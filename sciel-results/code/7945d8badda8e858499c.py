import time
import math
import zlib

_START_TIME = time.time()
_TIME_LIMIT = 300.0
_SAFETY_MARGIN = 2.0

# Spectral approach for Cap Sets in Z_3^n
# A set is a cap set if it contains no 3-term arithmetic progression.
# In Fourier analysis over Z_3^n, a random set has "flat" spectrum.
# Sets with large Fourier coefficients at non-zero frequencies are "structured" 
# and more likely to contain APs.
# We want to select vectors that look "random" or "uncorrelated" with existing structure.
# Since we don't know the existing set, we approximate by preferring vectors 
# that have low self-correlation or specific spectral properties.
# 
# However, a simpler and very effective proxy for "good" cap set members 
# in greedy heuristics is often related to the distribution of symbols.
# But the inspiration suggests using DFT.
# For a single vector v, its DFT is trivial. 
# The DFT is usually applied to the *indicator function* of the set.
# Since we are scoring individual vectors *before* they are added, 
# we can't compute the DFT of the current set easily without storing it.
#
# Alternative Spectral Interpretation:
# A vector v is "good" if it is orthogonal to many low-frequency patterns?
# Or, we can use the fact that vectors with balanced symbols (0,1,2 counts) 
# tend to have smaller L1 norm in their Fourier transform?
# Actually, the L2 norm of the DFT is constant (Parseval).
# The L1 norm of the DFT is minimized by random vectors.
# 
# Let's try a different structural change: 
# **Combinatorial Design / Orthogonal Array Heuristic**
# Instead of distance to core, we use a **hash-based diversity** metric 
# that mimics the properties of orthogonal arrays.
# We map the vector to a set of "features" based on linear combinations mod 3.
# If a vector shares too many features with previously high-priority vectors, 
# it is penalized.
#
# Let's stick to the Inspiration 1: Spectral Representation.
# How to compute a spectral score for a single vector v in Z_3^n?
# The "structure" of a vector can be measured by how "aligned" it is with 
# low-dimensional subspaces.
# A vector is "bad" if it lies in a small number of affine planes?
# 
# Let's try a simpler, robust structural change:
# **Adaptive Weighting of Symbol Counts based on Global Frequency**
# We track the global count of each symbol (0,1,2) across all vectors *seen so far* 
# (not just accepted, but seen). 
# We assume the "ideal" cap set has uniform symbol distribution.
# We penalize vectors that use symbols that are currently *over-represented* 
# in the high-priority pool.
# This is a form of **Negative Sampling** or **Diversity Maximization**.
#
# Wait, the current program already does "balance". 
# The difference: The current program balances *within the vector*. 
# The new strategy balances *across the stream*.
# If many high-priority vectors have lots of 0s, we should prioritize vectors 
# with fewer 0s (or more 1s/2s) to maintain global balance, assuming the 
# evaluator's greedy process will accept the diverse ones.
#
# Let's implement **Global Symbol Frequency Penalty**.

# State:
# - counts of 0, 1, 2 seen in vectors with high priority (top K)
# - We maintain a reservoir of top K vectors by a preliminary score
# - Preliminary score: random hash (to ensure diversity in the reservoir)
# - Final score: Base Score - Penalty based on deviation from uniform symbol distribution 
#   in the reservoir.

import random

# We need to be deterministic. Use a fixed seed for the hash.
# But we can use zlib.crc32 for a fast, deterministic hash.

_CORE_SIZE = 100
_core_vectors = []  # List of (score, vector)
_core_symbols = [0, 0, 0]  # Counts of 0, 1, 2 in the core

def _time_check():
    return (time.time() - _START_TIME) > (_TIME_LIMIT - _SAFETY_MARGIN)

def _preliminary_score(vector):
    """
    A fast, deterministic score to select candidates for the core.
    We use a hash to spread out the selection, ensuring the core represents 
    a diverse sample of the input space, not just the "best" by some metric.
    This allows us to estimate the global distribution of the "high priority" 
    region.
    """
    # Use CRC32 for speed and determinism
    # Convert vector to bytes for hashing
    # n is small, so this is fast.
    # We want a score in [0, 1)
    h = zlib.crc32(vector)
    return h / (2**32)

def priority(vector, n):
    if _time_check():
        # Fallback: return a simple deterministic hash to break ties
        # or a static balance score
        c0 = vector.count(0)
        c1 = vector.count(1)
        c2 = vector.count(2)
        ideal = n / 3.0
        dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
        return -dev + zlib.crc32(vector) / (2**32)

    # 1. Compute preliminary score to manage the core
    p_score = _preliminary_score(vector)
    
    # 2. Update Core
    # We want to keep the top _CORE_SIZE vectors by p_score.
    # Since p_score is random, this effectively maintains a random sample 
    # of size _CORE_SIZE from the stream.
    # This is O(K) per call if we use a list and sort occasionally, 
    # or a heap.
    # Given K=100 and potentially millions of calls, a heap is better.
    
    global _core_vectors, _core_symbols
    
    # We need to store the vector to update symbol counts if it's added.
    # But we also need to remove from symbol counts if it's removed.
    # This is expensive.
    
    # Alternative: 
    # Instead of a dynamic core that changes, we can use **Exponential Moving Average (EMA)** 
    # of symbol frequencies for vectors that pass a threshold.
    # Or, simpler: 
    # We assume the evaluator calls priority() for ALL vectors in some order.
    # If we assume the order is random, we can just track the global frequency 
    # of symbols in ALL vectors seen so far.
    # But the "high priority" vectors are the ones that will be accepted.
    # If our priority function is good, the accepted vectors will have high priority.
    # So, if we track the symbol distribution of vectors with *high* priority, 
    # we can bias against over-represented symbols.
    
    # Let's use a simpler state: 
    # A running average of symbol counts for the top 10% of preliminary scores.
    # This is hard to do online without storing the threshold.
    
    # Let's go back to the Heap approach but optimize the symbol count update.
    # We store (p_score, vector, symbol_counts_tuple) in the heap.
    # When we push, we add to _core_symbols.
    # When we pop, we subtract.
    
    import heapq
    
    # We need a min-heap of (p_score, vector).
    # But we also need to know the symbol counts to update _core_symbols.
    # We can compute symbol counts from the vector when needed, 
    # but that's O(n).
    # Let's store the counts in the heap item.
    # Heap item: (p_score, c0, c1, c2, vector)
    
    # Initialize global heap if not exists
    if not hasattr(priority, '_heap'):
        priority._heap = []
        priority._symbols = [0, 0, 0]
        
    heap = priority._heap
    symbols = priority._symbols
    
    # Compute symbol counts for this vector
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0: c0 += 1
        elif v == 1: c1 += 1
        else: c2 += 1
        
    # Compute final score
    # Base score: Balance of the vector itself (local)
    ideal = n / 3.0
    local_dev = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    base_score = -local_dev * 100.0
    
    # Global score: Penalty for using symbols that are over-represented in the core
    # If the core is empty, no penalty.
    # If the core has N vectors, the average count per vector is _core_symbols[i] / len(heap)
    # We want to minimize the contribution to the deviation from uniformity.
    # Let total_symbols = sum(_core_symbols)
    # Let expected_count = total_symbols / 3
    # If c_i is high, and _core_symbols[i] is already high, penalize.
    
    if heap:
        total_core_symbols = sum(symbols)
        if total_core_symbols > 0:
            avg_count = total_core_symbols / 3.0
            # Deviation of this vector's symbol usage from the "ideal" contribution
            # We want the *new* total to be more balanced.
            # New total symbols: symbols + [c0, c1, c2]
            # New total count: total_core_symbols + n
            # New ideal: (total_core_symbols + n) / 3
            # We want to minimize sum | (symbols[i] + c_i) - new_ideal |
            # This is a small optimization problem.
            # Since n is small, we can just compute the new deviation.
            
            new_total = total_core_symbols + n
            new_ideal = new_total / 3.0
            
            new_dev = 0
            for i in range(3):
                new_count = symbols[i] + [c0, c1, c2][i]
                new_dev += abs(new_count - new_ideal)
            
            # Compare with current deviation (if we added nothing, but we must add)
            # Actually, we just want to penalize if adding this vector *increases* 
            # the global imbalance significantly.
            # Or, simpler: reward vectors that *reduce* the global imbalance.
            
            # Let's compute the current global deviation (approx)
            # It's not strictly necessary to compute current, just the delta.
            # Let's use a simpler proxy: 
            # Penalty = sum( max(0, symbols[i] - avg_count) * c_i )
            # If symbol 0 is over-represented (symbols[0] > avg), and this vector has many 0s (c0 > 0), 
            # the penalty is high.
            
            penalty = 0.0
            for i in range(3):
                if symbols[i] > avg_count:
                    excess = symbols[i] - avg_count
                    # Scale by the vector's count of this symbol
                    penalty += excess * [c0, c1, c2][i]
            
            # Normalize penalty by n to make it comparable across dimensions
            penalty /= n
            base_score -= penalty * 10.0 # Weight factor
            
    # Update Heap
    # We want to keep the top _CORE_SIZE by p_score.
    # p_score is random, so this keeps a random sample.
    # Is a random sample the right "core"?
    # Ideally, the core should represent the *accepted* vectors.
    # Since we don't know acceptance, and p_score is random, 
    # this core is just a random sample of the input stream.
    # This might not be helpful if the input stream is biased.
    
    # Better Core Strategy:
    # Keep the top _CORE_SIZE vectors by *base_score* (before global penalty).
    # This approximates the set of "locally good" vectors.
    
    # Let's use base_score (local balance) as the heap key.
    # But base_score depends on n.
    
    # Let's refine the core update:
    # We push (base_score, c0, c1, c2, vector)
    # If heap size > _CORE_SIZE, pop the smallest base_score.
    
    if len(heap) < _CORE_SIZE:
        heapq.heappush(heap, (base_score, c0, c1, c2, vector))
        symbols[0] += c0
        symbols[1] += c1
        symbols[2] += c2
    else:
        if base_score > heap[0][0]:
            # Pop the worst
            _, pc0, pc1, pc2, _ = heapq.heapreplace(heap, (base_score, c0, c1, c2, vector))
            symbols[0] += c0 - pc0
            symbols[1] += c1 - pc1
            symbols[2] += c2 - pc2
        # else: do nothing
    
    return float(base_score)

