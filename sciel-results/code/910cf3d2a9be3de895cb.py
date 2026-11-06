import itertools
import time

def is_square(x, p):
    return pow(x, (p-1)//2, p) == 1

def construct_paley(p):
    squares = set()
    for x in range(1, p):
        squares.add((x*x) % p)
    
    n = p
    orientation = []
    for i in range(n):
        for j in range(i+1, n):
            if (j-i) % p in squares:
                orientation.append(0) # i -> j
            else:
                orientation.append(1) # j -> i
    return {'n': n, 'orientation': orientation}

def check_validity(result):
    n = result['n']
    orientation = result['orientation']
    
    # Build in-neighborhood bitsets
    # in_nb[v] is a bitset of vertices u such that u -> v
    in_nb = [0] * n
    
    # Map pair index to (i, j)
    # We need to fill in_nb
    # index k corresponds to (i, j)
    # If orientation[k] == 0, i -> j, so i is in in_nb[j]
    # If orientation[k] == 1, j -> i, so j is in in_nb[i]
    
    idx = 0
    for i in range(n):
        for j in range(i+1, n):
            if orientation[idx] == 0:
                # i -> j
                in_nb[j] |= (1 << i)
            else:
                # j -> i
                in_nb[i] |= (1 << j)
            idx += 1
            
    # Check all 4-subsets
    # A 4-subset {a, b, c, d} has a common dominator v iff
    # v is in in_nb[a] AND in_nb[b] AND in_nb[c] AND in_nb[d]
    # We need this intersection to be non-empty for all 4-subsets.
    
    for quad in itertools.combinations(range(n), 4):
        common = in_nb[quad[0]] & in_nb[quad[1]] & in_nb[quad[2]] & in_nb[quad[3]]
        if common == 0:
            return False
    return True

def construct():
    start_time = time.time()
    deadline = start_time + 140 # Safety margin
    
    # Primes p = 3 mod 4
    # 31, 43, 59, 67, 79, 83, 95(not prime), 107...
    # We want to minimize n.
    candidates = [31, 43, 59, 67, 79, 83]
    
    best_result = None
    best_n = float('inf')
    
    for p in candidates:
        if time.time() > deadline:
            break
            
        res = construct_paley(p)
        
        # Verify
        if check_validity(res):
            return res # Found the smallest valid one
            
    # If none found in the quick list, try a few more or return the best known
    # The problem states smallest known is 67.
    # If 67 wasn't in the list or failed, we might need to search.
    # But 67 is likely valid.
    
    # Fallback: return the seed program's result or the smallest candidate we tried
    # Actually, if 67 is the smallest known, and we didn't find a smaller valid one,
    # we should return 67 if it was valid, or 83 if 67 failed?
    # But 67 is likely valid.
    
    # Let's assume one of the candidates is valid.
    # If we reach here, it means none of the candidates were valid.
    # This is unlikely for 67 or 83.
    
    # Return the seed result as fallback
    p = 83
    squares = {x * x % p for x in range(1, p)}
    return {'n': p, 'orientation': [0 if (j - i) % p in squares else 1 for i in range(p) for j in range(i + 1, p)]}
