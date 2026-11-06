"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from functools import lru_cache
import sys

def synthesize(matrix, depth_limit):
    try:
        A = np.array(matrix, dtype=int)
    except Exception:
        A = np.array(matrix)
        A = (A != 0).astype(int)
    
    if A.ndim == 1:
        A = A.reshape(1, -1)
    
    m, n = A.shape
    
    # Convert rows to frozensets of indices for memoization
    rows = []
    for i in range(m):
        row = A[i]
        ones = np.where(row == 1)[0]
        rows.append(frozenset(int(x) for x in ones))
    
    gates = []
    next_wire = [n + 1]
    memo = {}
    
    # Initialize memo with inputs and zero
    for i in range(n):
        memo[frozenset([i])] = i
    memo[frozenset()] = n
    
    def get_subsets(s):
        """Yield all non-empty proper subsets of s."""
        # Only used for small sets
        lst = list(s)
        size = len(lst)
        # Iterate through bitmask
        for mask in range(1, (1 << size) - 1):
            sub = frozenset(lst[i] for i in range(size) if mask & (1 << i))
            yield sub

    @lru_cache(maxsize=None)
    def solve(s):
        """
        Returns the wire index for the linear form represented by frozenset s.
        Creates gates as needed.
        """
        if s in memo:
            return memo[s]
        
        size = len(s)
        if size == 0:
            return n
        if size == 1:
            return list(s)[0]
        
        best_cost = float('inf')
        best_split = None
        
        # Heuristic: If size is large, don't try all subsets.
        # Try a balanced split first.
        if size <= 12:
            # Try all splits
            for sub in get_subsets(s):
                # Avoid duplicate checks (sub and s^sub)
                if len(sub) > size // 2:
                    continue
                
                comp = s ^ sub
                # Both sub and comp must be non-empty
                if not sub or not comp:
                    continue
                
                w1 = solve(sub)
                w2 = solve(comp)
                
                # Cost is 1 gate
                # We assume the cost of sub and comp is already accounted for in their creation?
                # No, solve returns the wire. The gate is added here.
                # We want to minimize the number of NEW gates.
                # Since solve is memoized, the cost is 0 if already computed, 1+ if new.
                # This recursive approach naturally shares.
                
                # We just pick the first valid split? 
                # To minimize gates, we should pick the split that minimizes (cost(sub) + cost(comp)).
                # But we don't track cost separately. 
                # However, since we are building a tree, any split adds 1 gate.
                # The total gates is determined by the number of unique internal nodes in the DAG.
                # Any split leads to a valid DAG. 
                # To minimize total unique nodes, we want to maximize sharing.
                # Sharing happens if sub or comp is a common subexpression of other rows.
                # Greedy: Pick the split that produces the most "common" sub-expressions?
                # Simpler: Just pick a balanced split to keep depth low.
                # Depth limit is 16.
                # Balanced split ensures depth <= log2(n) + 1.
                # For n=16, depth <= 5.
                
                # Let's just use a deterministic balanced split to avoid complex cost tracking
                # which might be slow.
                # If we use balanced split, we might miss optimal sharing?
                # Yes, but it's a good heuristic.
                
                best_split = (sub, comp)
                break # Take first balanced-ish split? 
                # Actually, let's try to find a subset that is already in memo?
                # If sub is in memo, cost is 0 for that part.
                
                # Let's refine: 
                # 1. Check if any proper subset is in memo.
                # 2. If so, use it.
                # 3. Else, use balanced split.
                
                pass # Placeholder for logic below
        
        # Refined Logic:
        # 1. Check if any subset is already computed.
        found_subset = None
        if size <= 12:
            for sub in get_subsets(s):
                if sub in memo:
                    comp = s ^ sub
                    if comp in memo:
                        # Both available
                        w1 = memo[sub]
                        w2 = memo[comp]
                        # Create new gate
                        w_new = next_wire[0]
                        next_wire[0] += 1
                        gates.append([w1, w2])
                        memo[s] = w_new
                        return w_new
                    else:
                        # One available, one needs synthesis
                        found_subset = (sub, comp)
                        break
                elif comp in memo:
                    found_subset = (comp, sub)
                    break
        
        if found_subset:
            sub, comp = found_subset
            w1 = solve(sub)
            w2 = solve(comp)
            w_new = next_wire[0]
            next_wire[0] += 1
            gates.append([w1, w2])
            memo[s] = w_new
            return w_new
        
        # Fallback: Balanced split
        lst = list(s)
        mid = size // 2
        sub1 = frozenset(lst[:mid])
        sub2 = frozenset(lst[mid:])
        
        w1 = solve(sub1)
        w2 = solve(sub2)
        
        w_new = next_wire[0]
        next_wire[0] += 1
        gates.append([w1, w2])
        memo[s] = w_new
        return w_new

    outputs = []
    for r in rows:
        w = solve(r)
        outputs.append(w)
        
    return {'gates': gates, 'outputs': outputs}
# EVOLVE-BLOCK-END
