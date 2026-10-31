import time
import random
import math

# EVOLVE-BLOCK-START
_START = time.time()
_DEADLINE = _START + 110.0

def _score_fast(u):
    """
    Compute 1 + log(|U-U|/|U+U|) / log(2*max(u)+1).
    u must contain 0 and be nonnegative.
    """
    if not u:
        return 0.0
    m = max(u)
    if m < 1:
        return 0.0
    
    n = len(u)
    
    # Compute sumset U+U
    # Since u is sorted, we can try to be a bit smarter, but for n<100 sets are fine
    # Sumset elements are in [0, 2m]
    # Difference set elements are in [-m, m]
    
    sumset = set()
    diffset = set()
    
    for a in u:
        for b in u:
            sumset.add(a + b)
            diffset.add(a - b)
    
    len_s = len(sumset)
    len_d = len(diffset)
    
    if len_s == 0 or len_d == 0:
        return 0.0
        
    ratio = len_d / float(len_s)
    if ratio <= 0:
        return 0.0
        
    log_denom = math.log(2 * m + 1)
    if log_denom == 0:
        return 0.0
        
    return 1.0 + math.log(ratio) / log_denom

def _construct_dilated(n, B):
    """
    Construct a set using a dilated binary scheme.
    U is a subset of { sum_{i=0}^{k-1} d_i B^i } where d_i in {0, 1}.
    Here we generate all sums, but we might not use all.
    Actually, for the score, we want to maximize the ratio of diff/sum sizes relative to the log of the max element.
    
    Let U_k = { sum_{i=0}^{k-1} x_i B^i | x_i in {0, 1} }.
    |U_k| = 2^k.
    Max element is (B^k - 1) / (B - 1).
    Sumset: If B is large enough, representations are unique?
    If B > 2, the sum of two elements in U_k can be represented in base B with digits in {0, 1, 2}.
    Specifically, if B is odd, digits are 0, 1, 2.
    Number of sumset elements:
    The sum of two such numbers is a sum of 2k digits (each 0,1,2).
    Actually, it's easier to just generate and count for small k.
    """
    if B <= 1:
        B = 2
    
    # Generate the full set of sums
    # This set has 2^n elements.
    # If 2^n is too large, we can't evaluate it quickly.
    # n=20 -> 10^6 elements. 10^6 * 10^6 for pair sums is 10^12 -> too slow.
    # We need to estimate or use a smaller set.
    # Let's restrict n to 12 or 13 for direct evaluation in Python.
    # 2^12 = 4096. Pairs: 4096^2 = 16 million. This is borderline in Python (~5-10s).
    # 2^13 = 8192. Pairs: 67 million. Too slow.
    # So we stick to n <= 12 for the full set, or use a subset.
    
    # Let's try to generate the set directly.
    elems = [0]
    for _ in range(n):
        new_elems = []
        for e in elems:
            new_elems.append(e)
            new_elems.append(e + B**_ )
        elems = new_elems
        
    # Wait, B**_ is not the right way to build it.
    # Let's do it properly.
    elems = {0}
    for i in range(n):
        B_i = B ** i
        new_elems = set()
        for e in elems:
            new_elems.add(e)
            new_elems.add(e + B_i)
        elems = new_elems
        
    u = sorted(list(elems))
    return u

def construct():
    global _DEADLINE
    _DEADLINE = max(time.time(), _START + 110.0)
    
    best_u = [0, 1]
    best_s = _score_fast(best_u)
    
    random.seed(42)
    
    # Strategy 1: Dilated binary sets (Base B)
    # These have a specific structure.
    # Let's test bases 2, 3, 5, 7 with various depths.
    
    candidates = []
    
    # Fixed small sets
    candidates.append([0, 1, 2, 3, 4, 5])
    candidates.append([0, 1, 3, 6, 10, 15, 21]) # Triangular
    candidates.append([0, 2, 4, 6, 8, 10])
    
    # Dilated binary
    for B in [2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20]:
        for n in range(2, 13): # 2^12 = 4096 elements
            if time.time() > _DEADLINE - 10:
                break
            try:
                u = _construct_dilated(n, B)
                if len(u) > 5000: # Safety check
                    continue
                s = _score_fast(u)
                if s > best_s:
                    best_s = s
                    best_u = u
            except OverflowError:
                break
            except Exception:
                continue
    
    # Strategy 2: Local search on a few promising structures
    # We will take the best dilated set and try to tweak it.
    # Also try random sets with specific properties (e.g., convex, or sparse).
    
    # Let's run a short local search on the best found so far
    if time.time() < _DEADLINE - 5.0:
        # Simple local search: randomly change one element, add one, or remove one.
        u = list(best_u)
        cur_s = best_s
        m = max(u)
        
        # Limit iterations based on time
        max_iters = 1000
        iters = 0
        while iters < max_iters and time.time() < _DEADLINE - 2.0:
            iters += 1
            u2 = u[:]
            op = random.random()
            
            if op < 0.3 and len(u) > 1:
                # Remove random element (not 0)
                idx = random.randint(1, len(u)-1)
                u2.pop(idx)
            elif op < 0.6:
                # Add random element
                new_val = random.randint(1, int(m * 1.5))
                if new_val not in u2:
                    u2.append(new_val)
                    u2.sort()
            elif op < 0.9:
                # Change random element (not 0)
                idx = random.randint(1, len(u)-1)
                old_val = u2[idx]
                new_val = random.randint(1, int(m * 1.5))
                if new_val != old_val:
                    u2[idx] = new_val
                    if new_val in u2[:idx] or new_val in u2[idx+1:]:
                        continue
                    u2.sort()
            else:
                # Scale all elements by a factor? No, integers.
                # Try adding a large gap
                if m > 0:
                    u2.append(m + random.randint(1, m))
                    u2.sort()
            
            if len(u2) < 2 or 0 not in u2:
                continue
                
            s2 = _score_fast(u2)
            if s2 > cur_s:
                cur_s = s2
                u = u2
                m = max(u)
                
        if cur_s > best_s:
            best_s = cur_s
            best_u = u

    # Final check
    best_u = sorted(set(int(x) for x in best_u))
    if 0 not in best_u:
        best_u.append(0)
        
    return best_u
# EVOLVE-BLOCK-END
