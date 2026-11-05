import time
import heapq

_START = time.time()
_DEADLINE = 85.0

def _elapsed():
    return time.time() - _START

def construct():
    MOD = 101
    N = 10

    # Precompute modular inverses
    inv = [0] * (N + 1)
    for x in range(1, N + 1):
        inv[x] = pow(x, -1, MOD)

    # BFS to find minimum number of distinct denominators to reach each residue
    # State: (residue, used_mask)
    # We want to find for each residue r, the minimum k such that there exists
    # a subset of {1..10} of size k whose inverses sum to r mod 101.
    
    # dist[r] = minimum size found so far for residue r
    # parent[r] = (prev_residue, denominator_added) to reconstruct the path
    dist = [N + 1] * MOD
    parent = [None] * MOD  # parent[r] = (prev_r, x) where x is the denominator added
    
    # Start from residue 0 with empty set
    # Use BFS by size (since all edges have unit weight, BFS gives minimum size)
    # But we need to track which denominators have been used to ensure distinctness.
    # State: (size, residue, used_mask)
    # We want to minimize size for each residue.
    
    # Better approach: BFS where we explore by increasing size.
    # For each size k from 1 to N, we try all subsets of size k.
    # But that's what the original code does.
    
    # Alternative: Use Dijkstra-like approach where state is (residue, used_mask)
    # and we want to reach each residue with minimum popcount(used_mask).
    
    # Let's use a different approach: for each residue, we want the minimum subset size.
    # We can do BFS from (0, 0) where state is (residue, used_mask).
    # Transition: add denominator x not in used_mask, new_residue = (residue + inv[x]) % MOD
    # new_used_mask = used_mask | (1 << (x-1))
    
    # Priority queue: (size, residue, used_mask)
    # We want to find for each residue the minimum size.
    
    visited = {}  # (residue, used_mask) -> size
    pq = [(0, 0, 0)]  # (size, residue, used_mask)
    dist[r] = 0
    parent[r] = None
    
    best_size = [N + 1] * MOD
    best_mask = [0] * MOD
    best_size[0] = 0
    
    while pq:
        if _elapsed() > _DEADLINE - 2.0:
            break
        sz, r, mask = heapq.heappop(pq)
        
        if sz > best_size[r]:
            continue
        
        if (r, mask) in visited and visited[(r, mask)] <= sz:
            continue
        visited[(r, mask)] = sz
        
        if sz < best_size[r]:
            best_size[r] = sz
            best_mask[r] = mask
        
        # Try adding each denominator not yet used
        for x in range(1, N + 1):
            bit = 1 << (x - 1)
            if mask & bit:
                continue
            new_r = (r + inv[x]) % MOD
            new_mask = mask | bit
            new_sz = sz + 1
            
            if new_sz < best_size[new_r]:
                heapq.heappush(pq, (new_sz, new_r, new_mask))
    
    # Construct rows
    rows = []
    for r in range(MOD):
        if best_mask[r] == 0 and r != 0:
            # Fallback: find any valid subset
            # This shouldn't happen if all residues are reachable
            rows.append([1])
        else:
            mask = best_mask[r]
            if mask == 0 and r == 0:
                # Residue 0 with empty set - but we need non-empty list
                # Find smallest non-empty subset summing to 0
                found = False
                for x in range(1, N + 1):
                    if inv[x] % MOD == 0:
                        rows.append([x])
                        found = True
                        break
                if not found:
                    # Try pairs
                    for x in range(1, N + 1):
                        for y in range(x + 1, N + 1):
                            if (inv[x] + inv[y]) % MOD == 0:
                                rows.append([x, y])
                                found = True
                                break
                        if found:
                            break
                if not found:
                    rows.append([1])
            else:
                terms = [x for x in range(1, N + 1) if (mask >> (x - 1)) & 1]
                if not terms:
                    terms = [1]
                rows.append(terms)
    
    return rows
