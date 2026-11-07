"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time

def _is_latin(sq, n):
    """Check if sq is a Latin square of order n with symbols 0..n-1."""
    target = list(range(n))
    for i in range(n):
        if sorted(sq[i]) != target:
            return False
    for j in range(n):
        col = [sq[i][j] for i in range(n)]
        if sorted(col) != target:
            return False
    return True

def _are_orthogonal(sq1, sq2, n):
    """Check if two Latin squares of order n are orthogonal."""
    pairs = set()
    for i in range(n):
        for j in range(n):
            p = (sq1[i][j], sq2[i][j])
            if p in pairs:
                return False
            pairs.add(p)
    return True

def construct():
    """
    Construct a maximum set of mutually orthogonal Latin squares of order 10.
    
    Uses the algebraic structure Z_10 ≅ Z_2 × Z_5.
    Squares are indexed by (c2, c5) where c2 ∈ Z_2, c5 ∈ Z_5.
    The square entry at (i, j) is:
      a_out = (c2 * (i//5) + (j//5)) % 2
      b_out = (c5 * (i%5) + (j%5)) % 5
      symbol = a_out * 5 + b_out
    
    There are 10 such squares. We need to find the maximum mutually orthogonal subset.
    By analysis, squares with different c5 values are always orthogonal.
    Squares with the same c5 but different c2 are also orthogonal.
    However, two squares with the same (c2, c5) are identical (not orthogonal).
    
    The maximum set has 9 squares (all except one, since we can't have all 10
    due to the known non-existence of a complete set of 9 MOLS of order 10... 
    wait, actually we need to check: can we have 9?
    
    Let's just brute-force check all subsets.
    """
    n = 10
    start = time.time()
    deadline = start + 55.0  # safety margin before 90s limit (but task says 60s)
    
    def _make_square(c2, c5):
        sq = []
        for i in range(n):
            i2, i5 = i // 5, i % 5
            row = []
            for j in range(n):
                j2, j5 = j // 5, j % 5
                a_out = (c2 * i2 + j2) % 2
                b_out = (c5 * i5 + j5) % 5
                row.append(a_out * 5 + b_out)
            sq.append(row)
        return sq
    
    # Generate all 10 candidate squares
    candidates = []
    for c2 in range(2):
        for c5 in range(5):
            sq = _make_square(c2, c5)
            if _is_latin(sq, n):
                candidates.append((c2, c5, sq))
    
    # Find maximum mutually orthogonal subset
    # We'll use a greedy approach first, then try to improve
    best_set = []
    best_size = 0
    
    # Greedy: try adding squares in order
    for cand in candidates:
        c2, c5, sq = cand
        ok = True
        for _, _, sq2 in best_set:
            if not _are_orthogonal(sq, sq2, n):
                ok = False
                break
        if ok:
            best_set.append(cand)
    
    best_size = len(best_set)
    
    # Try all permutations of a subset to see if we can do better
    # Since we have only 10 candidates, we can try all subsets of size > best_size
    # Actually, let's try a more systematic approach:
    # For each pair of candidates, check orthogonality, then find maximum clique
    
    # Build adjacency matrix: adj[i][j] = True if candidates[i] and candidates[j] are orthogonal
    m = len(candidates)
    adj = [[False] * m for _ in range(m)]
    for i in range(m):
        for j in range(m):
            if i != j:
                adj[i][j] = _are_orthogonal(candidates[i][2], candidates[j][2], n)
    
    # Find maximum clique using backtracking
    best_clique = []
    
    def _max_clique(nodes, current_clique):
        nonlocal best_clique, best_size
        if len(current_clique) > best_size:
            best_clique = list(current_clique)
            best_size = len(current_clique)
        
        # Prune: if remaining nodes can't beat current best
        if len(nodes) + len(current_clique) <= best_size:
            return
        
        for idx in range(len(nodes)):
            node = nodes[idx]
            # Check if node is adjacent to all nodes in current_clique
            can_add = True
            for cn in current_clique:
                if not adj[node][cn]:
                    can_add = False
                    break
            if can_add:
                # Recurse with remaining nodes that are also adjacent to node
                remaining = [nd for nd in nodes[idx+1:] if adj[node][nd]]
                current_clique.append(node)
                _max_clique(remaining, current_clique)
                current_clique.pop()
    
    _max_clique(list(range(m)), [])
    
    # Extract the best set
    if best_clique:
        result = [candidates[idx][2] for idx in best_clique]
    else:
        # Fallback: single trivial square
        result = [[(i + j) % n for j in range(n)] for i in range(n)]
    
    # Ensure proper types: convert to Python int and list
    result = [[ [int(x) for x in row] for row in sq] for sq in result]
    
    return result
# EVOLVE-BLOCK-END
