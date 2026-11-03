"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random
import itertools

def _int_to_word(val, n, q):
    """Convert integer to list of digits in base q."""
    word = []
    for _ in range(n):
        word.append(val % q)
        val //= q
    return word

def _word_to_int(word, q):
    """Convert list of digits in base q to integer."""
    val = 0
    for i in range(len(word) - 1, -1, -1):
        val = val * q + word[i]
    return val

def _precompute_neighbors(q, n, radius=1):
    """Precompute neighbors for all words. Returns a list of lists of neighbor indices."""
    total = q ** n
    neighbors = [[] for _ in range(total)]
    
    # Precompute powers for conversion
    q_powers = [q ** i for i in range(n)]
    
    for w in range(total):
        word = _int_to_word(w, n, q)
        neigh = [w]
        if radius >= 1:
            for pos in range(n):
                orig = word[pos]
                for val in range(q):
                    if val == orig:
                        continue
                    word[pos] = val
                    neigh.append(_word_to_int(word, q))
                word[pos] = orig
        neighbors[w] = neigh
    return neighbors

def _tetracode_init(q, n):
    """
    Initialize with tetracode construction.
    Tetracode is a [4,2,3] code over GF(3).
    We extend it to length n by appending all possible tails.
    This yields 9 * 3^(n-4) codewords.
    For n=8, this is 9 * 3^4 = 9 * 81 = 729. 
    Wait, the best known is 486. 
    The tetracode construction for K_3(n,1) is often improved by partitioning the tails.
    Actually, the standard tetracode construction for K_3(8,1) yields 486 if we use a specific partitioning or if the tetracode itself is optimized.
    Let's check: 486 = 2 * 3^5 = 2 * 243.
    Or 486 = 6 * 3^4 = 6 * 81.
    
    The van Laarhoven construction uses a different method.
    Let's try a simpler greedy init if tetracode is too large or hard to prune.
    However, 486 is the target. A simple tetracode extension might be 729, which is worse.
    
    Let's use a greedy initialization which is fast and often close to optimal for small n.
    """
    return None # We will use greedy in construct

def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 115.0  # Leave 5s margin
    
    total_words = q ** n
    if total_words == 1:
        return [[0] * n]
        
    # Precompute neighbors
    neighbors = _precompute_neighbors(q, n, radius)
    
    # 1. Greedy Initialization
    # We want to cover all words.
    # Greedy: Pick the word that covers the most uncovered words.
    # To speed this up, we can use a "lazy" priority queue or just simple loops if n is small.
    # For n=8, q=3, total=6561. Greedy is O(total^2 * radius) which is ~ 6561^2 * 17 ~ 700M ops. Too slow in Python.
    
    # Better Init: Use the known 486 construction if possible, or a fast heuristic.
    # Let's try a "randomized greedy" or "partition based" init.
    
    # Actually, for K_3(8,1), the 486 code is well-studied.
    # A common construction is based on the tetracode.
    # The tetracode has 9 words.
    # If we take the tetracode and extend it, we get 9 * 3^(n-4) words.
    # For n=8, 9 * 81 = 729. This is larger than 486.
    # The improvement to 486 comes from a more clever extension or a different base code.
    
    # Let's use a simple but effective heuristic: 
    # Start with a random set of words, or use a "clustering" approach.
    # Or, simply use a greedy algorithm but optimized.
    
    # Given the time constraint, a pure Python greedy on 6561 words might be tight.
    # Let's try to generate a good initial solution using a "simplified tetracode" or just a random set and optimize.
    
    # Strategy: 
    # 1. Generate a random set of ~500 words.
    # 2. Use local search to minimize.
    
    # However, random start is risky. 
    # Let's try the "Greedy" but with a time limit. If it's too slow, fall back to random.
    
    # Actually, there is a known 486 code. Can we hardcode it? No, must be general.
    # But we can use a "structured" random start.
    
    # Let's try a "Greedy with Sampling" approach.
    # Or, better: Use the fact that the problem is small enough for a "Simulated Annealing" style local search from a random start.
    
    # Let's stick to a robust local search from a "Greedy" start, but make the Greedy fast.
    # We can approximate the greedy step by sampling a subset of candidates.
    
    covered = np.zeros(total_words, dtype=bool)
    codewords = []
    
    # Fast Greedy Initialization with Sampling
    # Instead of checking all words, check a random subset of size S each step.
    # S should be large enough to find a good candidate.
    sample_size = min(500, total_words)
    
    while not covered.all():
        if time.time() > deadline - 5.0:
            break
            
        best_word = -1
        best_gain = -1
        
        # Sample candidates
        candidates = random.sample(range(total_words), sample_size)
        
        for w in candidates:
            # Count uncovered neighbors
            gain = 0
            for u in neighbors[w]:
                if not covered[u]:
                    gain += 1
            if gain > best_gain:
                best_gain = gain
                best_word = w
            if best_gain == 0:
                break # No improvement possible from this sample? Unlikely unless covered.
                
        if best_word == -1:
            # Fallback: pick any uncovered word's neighbor
            uncovered_words = np.where(~covered)[0]
            if len(uncovered_words) == 0:
                break
            best_word = uncovered_words[0]
            
        codewords.append(best_word)
        for u in neighbors[best_word]:
            covered[u] = True
            
    # 2. Local Search to Minimize
    # Convert to list of indices
    cw_indices = codewords[:]
    
    # Compute coverage counts
    # count[w] = number of codewords covering w
    count = np.zeros(total_words, dtype=np.int32)
    for ci in cw_indices:
        for u in neighbors[ci]:
            count[u] += 1
            
    best_size = len(cw_indices)
    best_cw = cw_indices[:]
    
    # Local Search Loop
    while time.time() < deadline - 2.0:
        improved = False
        
        # Phase 1: Remove redundant codewords
        # A codeword ci is redundant if count[u] > 1 for all u in neighbors[ci]
        # We scan in random order to avoid bias.
        indices_to_check = list(range(len(cw_indices)))
        random.shuffle(indices_to_check)
        
        for i in indices_to_check:
            if i >= len(cw_indices):
                continue
            ci = cw_indices[i]
            # Check if redundant
            redundant = True
            for u in neighbors[ci]:
                if count[u] == 1:
                    redundant = False
                    break
            
            if redundant:
                # Remove ci
                for u in neighbors[ci]:
                    count[u] -= 1
                cw_indices.pop(i)
                improved = True
                if len(cw_indices) < best_size:
                    best_size = len(cw_indices)
                    best_cw = cw_indices[:]
        
        # Phase 2: Swap to improve
        # If no removals, try swapping.
        # Pick a random codeword ci.
        # Find a word cj that, if swapped, allows more removals or is "better".
        # A simple heuristic: Swap ci with cj if cj covers more "uniquely covered" words than ci does.
        # Or, just try random swaps and see if we can remove something afterwards.
        
        if not improved and cw_indices:
            # Try a few random swaps
            swaps_attempted = 0
            while swaps_attempted < 20 and time.time() < deadline - 2.0:
                swaps_attempted += 1
                i = random.randint(0, len(cw_indices) - 1)
                ci = cw_indices[i]
                
                # Pick a random candidate cj
                # To make it effective, pick cj from the neighbors of words that are "critically covered" (count==1) by ci.
                critical_words = [u for u in neighbors[ci] if count[u] == 1]
                
                if not critical_words:
                    # ci is redundant, should have been removed in Phase 1. 
                    # Maybe counts changed? Just remove it.
                    for u in neighbors[ci]:
                        count[u] -= 1
                    cw_indices.pop(i)
                    improved = True
                    if len(cw_indices) < best_size:
                        best_size = len(cw_indices)
                        best_cw = cw_indices[:]
                    break
                
                # Pick a random critical word
                u_crit = random.choice(critical_words)
                # Pick a random neighbor of u_crit as new candidate
                candidates_for_swap = neighbors[u_crit]
                random.shuffle(candidates_for_swap)
                
                for cj in candidates_for_swap:
                    if cj == ci:
                        continue
                    
                    # Check if swapping ci with cj is valid (i.e., doesn't uncover any word)
                    # New count = count - ball(ci) + ball(cj)
                    # We need min(new_count) >= 1
                    
                    # Optimization: Only check words in ball(ci) U ball(cj)
                    # But for simplicity and speed, we can just check if cj covers all critical words of ci.
                    # If cj covers all critical words, the swap is valid.
                    
                    valid = True
                    for u in critical_words:
                        if u not in neighbors[cj]:
                            valid = False
                            break
                    
                    if valid:
                        # Perform swap
                        for u in neighbors[ci]:
                            count[u] -= 1
                        for u in neighbors[cj]:
                            count[u] += 1
                        cw_indices[i] = cj
                        improved = True
                        break # Try to remove again in next iteration
                # End for cj
            # End while swaps
        # End while local search
        
    # Return best found
    result = []
    for ci in best_cw:
        result.append(_int_to_word(ci, n, q))
        
    return result

# EVOLVE-BLOCK-END
