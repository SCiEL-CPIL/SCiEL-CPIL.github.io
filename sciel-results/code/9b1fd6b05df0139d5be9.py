"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import sys

def _is_latin(sq):
    n = len(sq)
    for i in range(n):
        if sorted(sq[i]) != list(range(n)):
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != list(range(n)):
            return False
    return True

def _is_orthogonal_fast(sq1, sq2):
    """Fast orthogonality check using a 10x10 boolean grid."""
    n = len(sq1)
    seen = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            a = sq1[i][j]
            b = sq2[i][j]
            if seen[a][b]:
                return False
            seen[a][b] = True
    return True

def _generate_latin_candidates(n, max_candidates=200):
    """Generate candidate Latin squares using various methods."""
    candidates = []
    
    # Method 1: Linear squares L[i][j] = (a*i + b*j) % n where gcd(a,n)=1, gcd(b,n)=1
    for a in range(1, n):
        for b in range(1, n):
            if _gcd(a, n) == 1 and _gcd(b, n) == 1:
                sq = [[(a * i + b * j) % n for j in range(n)] for i in range(n)]
                if _is_latin(sq):
                    candidates.append(sq)
    
    # Method 2: Permutations of the standard cyclic square
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    rng = random.Random(12345)
    
    for _ in range(max_candidates - len(candidates)):
        # Random row permutation
        row_perm = list(range(n))
        rng.shuffle(row_perm)
        sq = [base[row_perm[i]] for i in range(n)]
        candidates.append(sq)
        
        # Random column permutation
        col_perm = list(range(n))
        rng.shuffle(col_perm)
        sq = [[base[i][col_perm[j]] for j in range(n)] for i in range(n)]
        candidates.append(sq)
        
        # Random symbol permutation
        sym_perm = list(range(n))
        rng.shuffle(sym_perm)
        sq = [[sym_perm[base[i][j]] for j in range(n)] for i in range(n)]
        candidates.append(sq)
        
        # Combined permutations
        row_perm = list(range(n))
        col_perm = list(range(n))
        sym_perm = list(range(n))
        rng.shuffle(row_perm)
        rng.shuffle(col_perm)
        rng.shuffle(sym_perm)
        sq = [[sym_perm[base[row_perm[i]][col_perm[j]]] for j in range(n)] for i in range(n)]
        candidates.append(sq)
    
    return candidates

def _gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def _construct_mols_10():
    """Construct MOLS of order 10 using greedy selection from candidate pool."""
    deadline = time.time() + 55  # Safety margin before 90s kill
    
    n = 10
    base = [[(i + j) % n for j in range(n)] for i in range(n)]
    squares = [base]
    
    # Generate a large pool of candidates
    candidates = _generate_latin_candidates(n, max_candidates=500)
    
    # Remove duplicates (by converting to tuple of tuples)
    seen = set()
    unique_candidates = []
    for sq in candidates:
        key = tuple(tuple(row) for row in sq)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(sq)
    
    # Greedy selection: for each candidate, check if orthogonal to all selected
    for sq in unique_candidates:
        if time.time() > deadline:
            break
        if len(squares) >= 9:
            break
        
        # Check orthogonality with all existing squares
        is_ortho_to_all = True
        for es in squares:
            if not _is_orthogonal_fast(sq, es):
                is_ortho_to_all = False
                break
        
        if is_ortho_to_all:
            squares.append(sq)
    
    # If we only have 1 square, try backtracking as fallback
    if len(squares) < 2 and time.time() < deadline - 5:
        new_sq = _find_orthogonal_square_bt(10, squares, deadline)
        if new_sq is not None and _is_latin(new_sq):
            if all(_is_orthogonal_fast(new_sq, es) for es in squares):
                squares.append(new_sq)
    
    return squares

def _find_orthogonal_square_bt(n, existing_squares, deadline):
    """Fallback backtracking search for an orthogonal square."""
    rng = random.Random(42)
    sq = [[-1] * n for _ in range(n)]
    
    row_used = [set() for _ in range(n)]
    col_used = [set() for _ in range(n)]
    ortho_used = [set() for _ in existing_squares]
    
    def get_candidates(i, j):
        cands = []
        for v in range(n):
            if v in row_used[i] or v in col_used[j]:
                continue
            ok = True
            for k, es in enumerate(existing_squares):
                e_val = es[i][j]
                if (e_val, v) in ortho_used[k]:
                    ok = False
                    break
            if ok:
                cands.append(v)
        return cands
    
    cells = [(i, j) for i in range(n) for j in range(n)]
    
    def backtrack(idx):
        if time.time() > deadline:
            return False
        if idx == n * n:
            return True
        
        best_ci = -1
        best_cands = None
        best_count = n + 1
        
        for ci in range(idx, n * n):
            i, j = cells[ci]
            if sq[i][j] >= 0:
                continue
            cands = get_candidates(i, j)
            if len(cands) < best_count:
                best_count = len(cands)
                best_cands = cands
                best_ci = ci
                if best_count <= 1:
                    break
        
        if best_ci == -1:
            return True
        
        if best_count == 0:
            return False
        
        i, j = cells[best_ci]
        cands = best_cands[:]
        rng.shuffle(cands)
        
        for v in cands:
            sq[i][j] = v
            row_used[i].add(v)
            col_used[j].add(v)
            pairs_added = []
            for k, es in enumerate(existing_squares):
                e_val = es[i][j]
                pairs_added.append((k, (e_val, v)))
                ortho_used[k].add((e_val, v))
            
            if backtrack(best_ci + 1):
                return True
            
            sq[i][j] = -1
            row_used[i].discard(v)
            col_used[j].discard(v)
            for k, pair in pairs_added:
                ortho_used[k].discard(pair)
        
        return False
    
    if backtrack(0):
        return [row[:] for row in sq]
    return None

_CACHE = None

def construct():
    global _CACHE
    if _CACHE is None:
        _CACHE = _construct_mols_10()
    result = []
    for sq in _CACHE:
        result.append([[int(v) for v in row] for row in sq])
    return result
# EVOLVE-BLOCK-END
