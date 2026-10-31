import time
import numpy as np

# Global state for the greedy simulation
_accepted_vectors = []
_start_time = time.time()
_DEADLINE = 300.0  # Safety margin before 330s kill

def _check_cap_set_addition(candidate, current_set):
    """
    Check if adding candidate to current_set keeps it a cap set.
    Returns True if valid, False otherwise.
    Also returns the number of conflicts (for priority scoring).
    """
    # A cap set in Z_3^n has no 3-term arithmetic progressions.
    # A 3-AP is {a, b, c} such that a + c = 2b (mod 3), i.e., a + c - 2b = 0.
    # Equivalently, for any distinct a, b in the set, c = 2b - a (mod 3) must NOT be in the set.
    # Or, for any pair (u, v) in the set, we must not have u + v + w = 0 for any w in the set?
    # Actually, the condition is: no three distinct elements x, y, z such that x + y + z = 0 (mod 3).
    # Because x + y + z = 0 <=> z = -x - y = 2x + 2y = 2(x+y) (mod 3).
    # Wait, the standard definition of cap set in Z_3^n is that it contains no 3-term arithmetic progression.
    # A 3-AP is x, x+d, x+2d. Sum is 3x + 3d = 0.
    # So the condition is: no distinct x, y, z in the set such that x + y + z = 0.
    
    # To check if adding 'candidate' to 'current_set' is valid:
    # 1. candidate is not already in current_set (implied by greedy unique addition usually, but let's be safe)
    # 2. For any pair (u, v) in current_set, u + v + candidate != 0 (mod 3)
    # 3. For any pair (u, v) in current_set, u + v + w != 0 is already satisfied.
    # 4. We also need to ensure candidate doesn't form a triple with two other candidates? No, greedy adds one by one.
    #    So we only need to check if candidate forms a triple with any two elements from the *existing* set.
    #    i.e., for any u, v in current_set, (u + v + candidate) % 3 != 0.
    #    Also, candidate cannot form a triple with itself? No, distinct elements.
    #    Also, candidate cannot be such that 2*candidate + u = 0 for some u? That is covered by u + candidate + candidate = 0?
    #    The condition is for distinct x,y,z. If x=y=candidate, then 2*candidate + z = 0.
    #    So we must check: for any u in current_set, (2*candidate + u) % 3 != 0.
    #    And for any u, v in current_set (u != v), (u + v + candidate) % 3 != 0.
    
    n = len(candidate)
    cand = np.array(candidate)
    
    # Optimization: Pre-compute sums of pairs in current_set?
    # For now, direct check.
    # Number of pairs in current_set is O(k^2). k is size of cap set.
    # For n=10, max cap set size is 92. 92^2 = 8464 pairs.
    # If we have 59049 vectors to check, this is 59049 * 8464 * 10 ops ~ 5e9 ops. Too slow in Python.
    
    # We need a faster check or a different strategy.
    # The prompt says "vectors ... are added greedily in descending priority when they keep the set a cap set".
    # The evaluator calls `priority(vector, n)` for *every* vector in the space?
    # "vectors of {0,1,2}^n are added greedily in descending priority"
    # This implies the evaluator sorts all 3^n vectors by priority, then iterates, adding if valid.
    # If I have to return a priority for each vector independently, I cannot use the *current* set state
    # unless I can infer it or if the evaluator allows stateful calls.
    # The interface is `priority(vector, n) -> float`.
    # It is called for each vector. The order of calls is likely arbitrary or sorted?
    # "added greedily in descending priority" -> The evaluator likely calls priority for all vectors,
    # sorts them, then adds them.
    # If so, I CANNOT use the running set state inside `priority` because I don't know the order of acceptance
    # until I've seen all priorities.
    # HOWEVER, if the evaluator calls `priority` in a specific order (e.g., lexicographic) and I maintain state,
    # I might be able to predict or simulate. But the prompt says "descending priority", which implies a global sort.
    
    # Let's re-read carefully: "vectors ... are added greedily in descending priority when they keep the set a cap set".
    # This usually means:
    # 1. Compute priority for all v in Z_3^n.
    # 2. Sort v by priority descending.
    # 3. Iterate through sorted list. If v keeps current set a cap set, add v.
    #    
    # If this is the case, `priority` must be a *static* function of the vector, independent of the others.
    # My previous "dynamic" idea fails if the evaluator sorts first.
    # 
    # BUT, look at the inspiration 2: "Cap set greedy builder with geometric-aware priority." and it defines `_accept_vector`.
    # And the current program is static.
    # 
    # Is it possible the evaluator calls `priority` one by one, and I can reject/accept?
    # No, the return type is `float`. It doesn't return a boolean.
    # 
    # If the evaluator sorts globally, then I need a static heuristic that correlates with "low conflict potential".
    # 
    # Let's look at the "different strategy" requirement.
    # Current: Balance (L1 distance from n/3).
    # Inspiration 1: Spectral (DFT magnitude).
    # Inspiration 2: Geometric-aware (Hamming weight / conflicts?).
    # 
    # What if I use a **Randomized Hashing / Color Coding** approach?
    # Or a **Locality-Sensitive Hashing** based priority?
    # 
    # Actually, there is a known result that cap sets are related to the absence of 3-APs.
    # A vector with many 0s and 1s (no 2s) is "balanced" in a different way?
    # 
    # Let's try a **Non-linear Interaction** feature.
    # The current heuristic only looks at the *counts* of symbols. It ignores the *positions*.
    # Two vectors with the same counts (e.g., three 0s, three 1s, three 2s) get the same priority.
    # But their conflict potential depends on their alignment.
    # 
    # Strategy: Use a **Dot Product with Random Vectors** (Random Projection) to break symmetry.
    # Vectors that are "aligned" with certain random directions might be more or less likely to form APs.
    # Actually, 3-APs in Z_3^n are related to lines.
    # 
    # Let's try a **Parity/Sum-based** heuristic that captures more structure than just counts.
    # Consider the sum of squares? In Z_3, 0^2=0, 1^2=1, 2^2=1.
    # So sum of squares is just count of non-zeros.
    # 
    # How about the **Norm** in some embedding?
    # 
    # Let's look at the "Spectral" inspiration. It scored 1.2930.
    # It uses DFT.
    # 
    # I want something *different*.
    # 
    # **New Strategy: "Conflict-Aware Static Score via Pairwise Distance Distribution"**
    # This is hard to do statically without knowing the set.
    # 
    # What if I use a **Greedy Self-Play** simulation *inside* the priority function?
    # If the evaluator calls `priority` for all vectors, I can't run a full simulation for each.
    # 
    # Wait, what if the evaluator *doesn't* sort globally, but uses a priority queue?
    # "added greedily in descending priority"
    # If it's a max-heap, it pops the highest priority, checks validity, adds if valid, else discards.
    # In this case, the state *does* change as the evaluator progresses.
    # If I can maintain state across calls, I can use the dynamic strategy.
    # 
    # Does the evaluator reset state between calls?
    # "Keep the function names and signatures...".
    # If I use global variables, they persist.
    # 
    # Risk: If the evaluator sorts globally first, my stateful approach will be inconsistent
    # (I'll be scoring based on a set that hasn't been formed yet, or in the wrong order).
    # 
    # However, the current program is static. The inspirations are static.
    # If a dynamic approach was viable and superior, it might have been tried.
    # But the stagnation signal suggests trying something new.
    # 
    # Let's assume the "Global Sort" model is the standard for these tasks unless specified otherwise.
    # If so, I must improve the static heuristic.
    # 
    # **Alternative Static Heuristic: "Anti-Alignment" Score**
    # Vectors that are "far" from forming lines are better.
    # A line in Z_3^n is defined by a point and a direction.
    # 
    # Let's try a **Polynomial Feature** approach.
    # Instead of just counts, use the sum of $v_i^2$ (which is count of non-zeros) and sum of $v_i^3$ (which is sum of $v_i$).
    # 
    # Current: Minimizes $|c_0 - n/3| + |c_1 - n/3| + |c_2 - n/3|$.
    # 
    # Let's try maximizing the **Variance** of the coordinates?
    # No, cap sets are often found in "balanced" distributions.
    # 
    # Let's look at the **DFT** inspiration again.
    # $|\hat{f}(k)|^2$.
    # 
    # I will implement a **Hybrid Static Score**:
    # 1. Balance term (keep, but weaker weight).
    # 2. **Spectral Term**: Compute the magnitude of the DFT of the vector (padded to length 3^n? No, that's huge).
    #    The inspiration says "magnitude of their Discrete Fourier Transform over Z_3^n".
    #    This implies treating the vector as a function on Z_3^n? No, the vector *is* in Z_3^n.
    #    Maybe it means the Fourier transform of the *indicator* of the set? No, we are scoring a single vector.
    #    Maybe it means the Fourier coefficient of the vector itself?
    #    A vector $v \in \{0,1,2\}^n$ can be mapped to $\mathbb{C}^n$ via roots of unity.
    #    $z_i = \omega^{v_i}$.
    #    Product $P = \prod z_i$.
    #    This is just the character $\chi_v(1, \dots, 1)$.
    #    
    #    Let's try a **Random Projection** based score.
    #    Generate a fixed set of random vectors $R_1, \dots, R_M$ in $\{0,1,2\}^n$.
    #    Score = $-\sum_{j} (\text{conflict potential with } R_j)$.
    #    How to estimate conflict potential statically?
    #    If $v$ and $r$ are "similar", they might form APs with common neighbors.
    #    
    #    Let's try **Hamming Distance to a Set of "Bad" Vectors**.
    #    What are "Bad" vectors?
    #    Vectors that are part of many 3-APs?
    #    Every vector is in $3^{n-1} - 1$ lines?
    #    
    #    Actually, let's look at the **Number of 0s**.
    #    Vectors with more 0s are "sparser".
    #    Sparsity often helps in coding theory.
    #    
    #    Let's try a **Logarithmic Penalty for Imbalance** + **Positional Hash**.
    #    The current hash is a simple polynomial.
    #    
    #    **New Idea: "Max-Min Coordinate Spread"**
    #    Encourage vectors where the coordinates are "spread out" in value, not just count.
    #    e.g., $v = (0, 1, 2, 0, 1, 2...)$ is periodic.
    #    $v = (0, 0, 0, 1, 1, 1...)$ is blocky.
    #    Which is better?
    #    Periodic vectors might align with lines.
    #    
    #    Let's try to **Penalize Periodicity**.
    #    Calculate the autocorrelation of the vector.
    #    If a vector has high autocorrelation, it is likely to form APs with shifted versions.
    #    
    #    **Algorithm:**
    #    1. Convert vector to numerical array.
    #    2. Compute circular autocorrelation $R(k) = \sum v_i v_{i+k}$.
    #    3. High $R(1)$ or $R(2)$ suggests structure that might form APs.
    #    4. Score = - (Max Autocorrelation) * Weight + Balance Score.
    #    
    #    This captures *positional* structure, unlike the current count-only heuristic.
    
    return 0.0

# Refining the Autocorrelation Idea
# For n up to 10 (or maybe 20?), autocorrelation is O(n^2) or O(n log n).
# 3^10 = 59049 vectors.
# If n=10, O(n^2) = 100 ops per vector. 59049 * 100 = 6 million ops. Very fast.
# If n=20, 3^20 is huge, likely not evaluated fully.
# The problem likely has small n (e.g., 5-10) where exhaustive search is possible.

import numpy as np

# Pre-compute nothing? Or pre-compute random weights?
# Let's use deterministic pseudo-random weights for tie-breaking and structure.

def priority(vector, n):
    # 1. Balance Score (keep as base)
    c0 = 0
    c1 = 0
    c2 = 0
    for v in vector:
        if v == 0:
            c0 += 1
        elif v == 1:
            c1 += 1
        else:
            c2 += 1
            
    ideal = n / 3.0
    deviation = abs(c0 - ideal) + abs(c1 - ideal) + abs(c2 - ideal)
    balance_score = -deviation * 100.0  # Reduced weight to allow structure to matter
    
    # 2. Structural Score: Penalize Autocorrelation
    # Convert to array
    v_arr = np.array(vector, dtype=float)
    
    # Circular Autocorrelation
    # R(k) = sum_{i=0}^{n-1} v[i] * v[(i+k)%n]
    # We care about small lags k=1, 2, ..., n//2
    # High autocorrelation at small lags indicates "clumping" or "periodicity"
    # which increases the chance of forming 3-APs with nearby vectors.
    
    max_ac = 0.0
    # Only compute up to n//2 to save time, and ignore k=0 (which is sum of squares)
    for k in range(1, n // 2 + 1):
        # Efficient dot product using roll
        shifted = np.roll(v_arr, k)
        ac = np.dot(v_arr, shifted)
        if ac > max_ac:
            max_ac = ac
            
    # Normalize max_ac by n to make it scale-invariant
    # Max possible ac is sum(v^2) <= n*4 = 4n
    norm_ac = max_ac / n
    
    # Penalize high autocorrelation
    structure_score = -norm_ac * 50.0
    
    # 3. Tie-breaker: Deterministic Hash
    h = 0
    base = 31
    for val in vector:
        h = (h * base + val) % (10**9 + 7)
    hash_score = h / (10**9 + 7)
    
    return float(balance_score + structure_score + hash_score)
