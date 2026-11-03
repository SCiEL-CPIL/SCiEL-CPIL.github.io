import time
import random
import sys

# Precomputation for q=3, n=7
Q = 3
N = 7
R = 1
TOTAL_WORDS = Q ** N  # 2187

# neighbors[w] = list of words covered by codeword w (distance <= 1)
# coverers[w] = list of codewords that cover word w (distance <= 1)
# These are the same lists essentially, just transposed.
# Since the graph is regular (each word is within dist 1 of 15 words), 
# neighbors[w] and coverers[w] are identical sets.

def precompute_neighbors():
    neighbors = [[] for _ in range(TOTAL_WORDS)]
    for w in range(TOTAL_WORDS):
        # Convert to base-3 tuple
        digits = [0] * N
        temp = w
        for i in range(N):
            digits[i] = temp % Q
            temp //= Q
        
        # Add self
        neighbors[w].append(w)
        
        # Add neighbors at distance 1
        for pos in range(N):
            orig = digits[pos]
            for new_val in range(Q):
                if new_val == orig:
                    continue
                digits[pos] = new_val
                # Convert back to int
                val = 0
                mult = 1
                for i in range(N):
                    val += digits[i] * mult
                    mult *= Q
                neighbors[w].append(val)
            digits[pos] = orig
            
    return neighbors

# Global precomputation
NEIGHBORS = precompute_neighbors()
# COVERERS is the same as NEIGHBORS because the relation is symmetric
COVERERS = NEIGHBORS

def greedy_cover(random_seed=None):
    if random_seed is not None:
        random.seed(random_seed)
    
    uncovered = set(range(TOTAL_WORDS))
    code = []
    
    # To speed up, we can maintain a list of candidate codewords
    # and their current coverage count.
    # However, simple greedy is:
    # 1. Find c with max |neighbors[c] & uncovered|
    # 2. Add c, remove neighbors[c] from uncovered
    
    # Optimization: 
    # Instead of set intersection every time, we can keep a count.
    # But N is small (2187). 2187 * 15 is ~30k ops. 
    # If we do this 180 times, it's 5M ops. Fast enough.
    
    while uncovered:
        best_c = -1
        max_cov = 0
        
        # Randomize iteration order to break ties randomly
        candidates = list(range(TOTAL_WORDS))
        random.shuffle(candidates)
        
        for c in candidates:
            # Count how many uncovered words are covered by c
            cov = 0
            for w in NEIGHBORS[c]:
                if w in uncovered:
                    cov += 1
            
            if cov > max_cov:
                max_cov = cov
                best_c = c
                
        if best_c == -1:
            break # Should not happen
            
        code.append(best_c)
        for w in NEIGHBORS[best_c]:
            uncovered.discard(w)
            
    return code

def local_search(code, time_limit=None):
    if time_limit is None:
        time_limit = 10.0 # seconds per local search run
        
    start_time = time.time()
    code_set = set(code)
    
    # 1. 1-Reduction: Remove redundant codewords
    # We need to track cover counts
    cover_count = [0] * TOTAL_WORDS
    for c in code:
        for w in NEIGHBORS[c]:
            cover_count[w] += 1
            
    changed = True
    while changed:
        changed = False
        # Iterate in reverse to allow removals
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check if removing c leaves all cover_count >= 1
            # Temporarily decrement
            is_redundant = True
            for w in NEIGHBORS[c]:
                cover_count[w] -= 1
                if cover_count[w] == 0:
                    is_redundant = False
                    # Restore and break
                    cover_count[w] += 1
                    break
            
            if is_redundant:
                # Check if any other word became 0? 
                # We broke early if one became 0. 
                # If we didn't break, all were >= 1 before, so >= 0 now.
                # Wait, if we didn't break, it means all were >= 1 after decrement?
                # No, the loop breaks if count becomes 0.
                # So if is_redundant is True, it means we finished the loop without finding a 0.
                # Thus all counts are >= 1.
                code.pop(i)
                code_set.discard(c)
                changed = True
            else:
                # Restore the last one we decremented? 
                # No, we restored it inside the break.
                # But what about the ones decremented before the break?
                # We need to restore ALL of them.
                # My logic above was flawed.
                
                # Let's restart this part with a safer check.
                pass
                
    # Let's implement a cleaner 1-reduction
    # Re-initialize cover_count
    cover_count = [0] * TOTAL_WORDS
    for c in code:
        for w in NEIGHBORS[c]:
            cover_count[w] += 1
            
    changed = True
    while changed:
        changed = False
        for i in range(len(code) - 1, -1, -1):
            c = code[i]
            # Check min count in neighbors[c]
            min_count = min(cover_count[w] for w in NEIGHBORS[c])
            if min_count > 1:
                # Safe to remove
                for w in NEIGHBORS[c]:
                    cover_count[w] -= 1
                code.pop(i)
                code_set.discard(c)
                changed = True
            elif min_count == 1:
                # Cannot remove directly. 
                # (We don't do swaps here, just removals)
                pass
            # If min_count == 0, code is invalid, shouldn't happen
            
    # 2. 2-1 Exchange
    # Try to replace 2 codewords with 1
    # Iterate until no improvement or time limit
    while time.time() - start_time < time_limit:
        improved = False
        # Shuffle code to randomize search order
        random.shuffle(code)
        
        for i in range(len(code)):
            if time.time() - start_time > time_limit:
                break
            c1 = code[i]
            for j in range(i + 1, len(code)):
                if time.time() - start_time > time_limit:
                    break
                c2 = code[j]
                
                # Temporarily remove c1 and c2
                # Decrement counts
                # To avoid modifying global state and reverting, 
                # we can calculate the "deficit"
                
                # Words covered ONLY by c1 or c2 (or both) will have count reduced.
                # We need to find words that become 0.
                
                # Efficient check:
                # Create a temporary count array? Too slow.
                # Instead, identify the set of words that depend on c1 or c2.
                
                # Let's just use the cover_count array and modify/revert.
                # It's O(30) per pair. 30 * 32000 = 1M ops. Fine.
                
                # Save original counts for affected words
                affected = set()
                for w in NEIGHBORS[c1]:
                    affected.add(w)
                for w in NEIGHBORS[c2]:
                    affected.add(w)
                    
                original_counts = {w: cover_count[w] for w in affected}
                
                # Decrement
                for w in NEIGHBORS[c1]:
                    cover_count[w] -= 1
                for w in NEIGHBORS[c2]:
                    cover_count[w] -= 1
                    
                # Find critical words (count == 0)
                critical = []
                for w in affected:
                    if cover_count[w] == 0:
                        critical.append(w)
                        
                if not critical:
                    # We removed 2 and didn't break anything. 
                    # This is a 2-0 move.
                    code.pop(j)
                    code.pop(i)
                    code_set.discard(c1)
                    code_set.discard(c2)
                    improved = True
                    break # Restart outer loops as indices changed
                    
                else:
                    # We need to find a c3 that covers all critical words
                    # c3 must be in the intersection of COVERERS[w] for w in critical
                    # And c3 must not be in code_set
                    
                    # Start with all possible codewords
                    # Better: Start with candidates from the first critical word
                    candidates = set(COVERERS[critical[0]])
                    for w in critical[1:]:
                        candidates.intersection_update(COVERERS[w])
                        
                    # Remove current code from candidates
                    candidates -= code_set
                    
                    if candidates:
                        # Pick one (random or first)
                        c3 = random.choice(list(candidates))
                        # Perform swap
                        code.pop(j)
                        code.pop(i)
                        code_set.discard(c1)
                        code_set.discard(c2)
                        code.append(c3)
                        code_set.add(c3)
                        
                        # Update cover counts
                        # We already decremented c1, c2.
                        # Now increment c3
                        for w in NEIGHBORS[c3]:
                            cover_count[w] += 1
                            
                        improved = True
                        break # Restart outer loops
                    else:
                        # No valid swap, revert counts
                        for w in affected:
                            cover_count[w] = original_counts[w]
                        
            if improved:
                break
                
        if not improved:
            break
            
    return code

def construct(q, n, radius):
    # Task is fixed to q=3, n=7, radius=1
    if q != 3 or n != 7 or radius != 1:
        # Fallback to seed logic or similar for other cases, 
        # but task contract implies this specific instance.
        # We'll just return a trivial valid code if params differ, 
        # though unlikely.
        pass
        
    start_time = time.time()
    deadline = start_time + 115.0 # 5s margin
    
    best_code = None
    best_size = float('inf')
    
    seed_counter = 0
    
    # First run: Deterministic greedy (seed 0)
    code = greedy_cover(0)
    code = local_search(code, time_limit=10.0)
    best_code = code
    best_size = len(code)
    
    # Subsequent runs: Randomized greedy
    while time.time() < deadline:
        seed_counter += 1
        code = greedy_cover(seed_counter)
        # Allocate remaining time to local search, min 1s, max 10s
        remaining = deadline - time.time()
        ls_time = max(1.0, min(10.0, remaining * 0.8))
        code = local_search(code, time_limit=ls_time)
        
        if len(code) < best_size:
            best_code = code
            best_size = len(code)
            
    return best_code

# Ensure global precomputation happens once
# But construct might be called multiple times? 
# "Each evaluation must finish within 120s". 
# Usually construct is called once per evaluation.
# If called multiple times, precomputation is cached.

# To be safe, move precomputation into construct or use a flag
_precomputed = False
_NEIGHBORS = None
_COVERERS = None

def _ensure_precomputed():
    global _precomputed, _NEIGHBORS, _COVERERS
    if not _precomputed:
        _NEIGHBORS = precompute_neighbors()
        _COVERERS = _NEIGHBORS
        _precomputed = True

def construct(q, n, radius):
    _ensure_precomputed()
    # ... rest of logic using _NEIGHBORS, _COVERERS ...
    
    # Re-write logic using globals
    pass

# Final assembly
