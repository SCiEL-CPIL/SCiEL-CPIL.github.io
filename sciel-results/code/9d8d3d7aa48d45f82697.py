import time
import itertools

P = {'n': 10}

def _latin_ok(L, n):
    for i in range(n):
        if sorted(L[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [L[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _orth(L1, L2, n):
    seen = set()
    for i in range(n):
        for j in range(n):
            pr = (L1[i][j], L2[i][j])
            if pr in seen:
                return False
            seen.add(pr)
    return True

def _isotopic(L1, L2):
    # Check if L1 and L2 are isotopic: L1[i][j] == L2[pi(i)][pj(j)] for some pi, pj
    # This is O(n^2) in the number of permutations, which is too slow for n=10 (10! = 3.6M)
    # However, we can use a signature based on the multiset of row multisets and column multisets,
    # or more simply, just check a few invariants.
    # A cheaper check: if the sorted list of sorted rows is the same, and sorted list of sorted cols is the same.
    rows1 = sorted(tuple(sorted(row)) for row in L1)
    rows2 = sorted(tuple(sorted(row)) for row in L2)
    if rows1 != rows2:
        return False
    cols1 = sorted(tuple(sorted(L1[i][j] for i in range(len(L1)))) for j in range(len(L1[0])))
    cols2 = sorted(tuple(sorted(L2[i][j] for i in range(len(L2)))) for j in range(len(L2[0])))
    if cols1 != cols2:
        return False
    # This is a necessary condition, not sufficient, but good enough for pruning.
    # Full isotopy check is expensive. We'll rely on this + exact pair match if needed.
    # Actually, for the greedy pool, we want distinct squares.
    # Two squares that are isotopic might still be orthogonal?
    # No, if L2 is a symbol permutation of L1, they are not orthogonal (pairs are (x, p(x)), which are distinct for each x, but wait...
    # If L2[i][j] = p(L1[i][j]), then the pair is (x, p(x)). As x ranges 0..9, p(x) ranges 0..9.
    # The pairs are (0, p(0)), ..., (9, p(9)). There are only 10 distinct pairs.
    # But we need 100 distinct pairs. So they are NOT orthogonal.
    # Similarly for row/col perms.
    # So we can safely skip isotopic squares.
    # The check above is fast.
    return True

def construct():
    n = P['n']
    t0 = time.time()
    deadline = 55.0  # safety margin under 60s

    # We know that for n=10, there exist 9 MOLS.
    # The group Z_10 is abelian. The cyclic Latin square L[i][j] = (i+j)%10 is one.
    # Are there others in the same isotopy class?
    # The number of MOLS orthogonal to a given cyclic Latin square of order n
    # depends on the structure.
    # For n=10, it is known that the maximum number of MOLS is 9.
    # However, finding them via search is hard.
    # But maybe we can find a few via a smarter search.
    
    # Let's try to generate a pool of candidates from the "linear" family over Z_10.
    # L[i][j] = (a*i + b*j) % 10 is a Latin square iff gcd(a, 10)=1 and gcd(b, 10)=1.
    # Wait, is that true?
    # Row i: a*i + b*j. As j varies, b*j varies. If gcd(b,10)=1, b*j is a permutation of 0..9.
    # So row is a permutation.
    # Col j: a*i + b*j. As i varies, a*i varies. If gcd(a,10)=1, a*i is a permutation.
    # So col is a permutation.
    # So any a, b with gcd(a,10)=1 and gcd(b,10)=1 gives a Latin square.
    # The number of such (a,b) pairs is 4*4 = 16.
    # These 16 squares are all in the same isotopy class as (i+j)%10?
    # L[i][j] = a*i + b*j.
    # If we permute rows by i' = a^{-1} i, then L[i'][j] = i' + b*j.
    # If we permute cols by j' = b^{-1} j, then L[i][j'] = i + j'.
    # So yes, they are all isotopic to (i+j)%10.
    # And as noted, isotopic squares are not orthogonal to each other.
    # So this family gives only 1 square in any orthogonal set.
    
    # We need squares from different isotopy classes.
    # The next best thing is to search for Latin squares orthogonal to the first one.
    # Let's implement a backtracking search for the second square.
    
    # Base square: S0[i][j] = (i+j)%10
    S0 = [[(i+j)%n for j in range(n)] for i in range(n)]
    
    # We want to find S1 such that (S0[i][j], S1[i][j]) are all distinct.
    # This is equivalent to finding a Latin square S1 such that for each symbol pair (x,y),
    # there is exactly one cell (i,j) with S0[i][j]=x and S1[i][j]=y.
    # This is a system of distinct representatives or exact cover.
    
    # Let's use a simple backtracking with constraint propagation.
    # State: S1 grid, initially all -1.
    # Constraints:
    # 1. Each row is a permutation of 0..9.
    # 2. Each col is a permutation of 0..9.
    # 3. For each (x,y) in 0..9 x 0..9, there is exactly one (i,j) with S0[i][j]=x and S1[i][j]=y.
    
    # We can iterate over cells in a specific order.
    # For each cell, the possible values are those not used in the row, not used in the col,
    # and such that the pair (S0[i][j], val) is not already used.
    
    # To speed up, we can use the most constrained variable (fewest options) first.
    # But for simplicity, let's try row-major order first.
    
    def solve_mols(S0, n):
        S1 = [[-1]*n for _ in range(n)]
        row_used = [set() for _ in range(n)]
        col_used = [set() for _ in range(n)]
        pair_used = set()
        
        # Precompute options for each cell
        # options[i][j] = list of possible values
        # But this changes as we fill.
        
        # We'll use recursion with backtracking.
        # To avoid deep recursion limits, we can use an explicit stack or just hope it's not too deep.
        # Depth is 100. Python recursion limit is usually 1000, so it's fine.
        
        # Optimization: If a cell has only one option, fill it immediately (unit propagation).
        
        def is_valid(i, j, val):
            if val in row_used[i]:
                return False
            if val in col_used[j]:
                return False
            if (S0[i][j], val) in pair_used:
                return False
            return True
        
        def backtrack(pos):
            if pos == n*n:
                return True
            
            # Find next unfilled cell
            # For simplicity, we stick to row-major order, but we could pick the most constrained one.
            # Let's try to pick the cell with the fewest available values.
            min_options = None
            min_count = 11
            for r in range(n):
                for c in range(n):
                    if S1[r][c] == -1:
                        count = 0
                        for v in range(n):
                            if is_valid(r, c, v):
                                count += 1
                        if count < min_count:
                            min_count = count
                            min_options = (r, c)
                        if count == 0:
                            return False # Dead end
            
            if min_options is None:
                return True # All filled
            
            r, c = min_options
            for v in range(n):
                if is_valid(r, c, v):
                    S1[r][c] = v
                    row_used[r].add(v)
                    col_used[c].add(v)
                    pair_used.add((S0[r][c], v))
                    
                    if backtrack(pos + 1):
                        return True
                    
                    S1[r][c] = -1
                    row_used[r].remove(v)
                    col_used[c].remove(v)
                    pair_used.remove((S0[r][c], v))
            
            return False
        
        if backtrack(0):
            return S1
        else:
            return None
    
    # Let's try to find the second square.
    # If the search is too slow, we might not find it in time.
    # But let's try.
    
    t_start_search = time.time()
    S1 = solve_mols(S0, n)
    
    if S1 is not None:
        # Verify
        if _latin_ok(S1, n) and _orth(S0, S1, n):
            squares = [S0, S1]
        else:
            squares = [S0]
    else:
        squares = [S0]
    
    # Now try to find a third square orthogonal to S0 and S1.
    if len(squares) == 2:
        # We need S2 orthogonal to S0 and S1.
        # This means pairs (S0[i][j], S2[i][j]) are distinct AND (S1[i][j], S2[i][j]) are distinct.
        # Or more generally, for any pair of existing squares, the pairs must be distinct.
        # Let's modify the solver to take a list of existing squares.
        
        def solve_mols_multi(existing, n):
            S_new = [[-1]*n for _ in range(n)]
            row_used = [set() for _ in range(n)]
            col_used = [set() for _ in range(n)]
            # pair_used for each existing square
            pair_used = [set() for _ in existing]
            
            def is_valid(i, j, val):
                if val in row_used[i]:
                    return False
                if val in col_used[j]:
                    return False
                for idx, S in enumerate(existing):
                    if (S[i][j], val) in pair_used[idx]:
                        return False
                return True
            
            def backtrack(pos):
                if pos == n*n:
                    return True
                
                min_options = None
                min_count = 11
                for r in range(n):
                    for c in range(n):
                        if S_new[r][c] == -1:
                            count = 0
                            for v in range(n):
                                if is_valid(r, c, v):
                                    count += 1
                            if count < min_count:
                                min_count = count
                                min_options = (r, c)
                            if count == 0:
                                return False
                
                if min_options is None:
                    return True
                
                r, c = min_options
                for v in range(n):
                    if is_valid(r, c, v):
                        S_new[r][c] = v
                        row_used[r].add(v)
                        col_used[c].add(v)
                        for idx, S in enumerate(existing):
                            pair_used[idx].add((S[r][c], v))
                        
                        if backtrack(pos + 1):
                            return True
                        
                        S_new[r][c] = -1
                        row_used[r].remove(v)
                        col_used[c].remove(v)
                        for idx, S in enumerate(existing):
                            pair_used[idx].remove((S[r][c], v))
                
                return False
            
            if backtrack(0):
                return S_new
            else:
                return None
        
        if time.time() - t0 < deadline - 5:
            S2 = solve_mols_multi(squares, n)
            if S2 is not None:
                if _latin_ok(S2, n) and all(_orth(S, S2, n) for S in squares):
                    squares.append(S2)
        
        # Try for a fourth
        if len(squares) == 3 and time.time() - t0 < deadline - 5:
            S3 = solve_mols_multi(squares, n)
            if S3 is not None:
                if _latin_ok(S3, n) and all(_orth(S, S3, n) for S in squares):
                    squares.append(S3)
    
    # Ensure we have at least 1
    if not squares:
        squares = [S0]
        
    # Convert to pure Python ints and lists
    result = [[[int(v) for v in row] for row in L] for L in squares]
    return result
