"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from itertools import combinations

# EVOLVE-BLOCK-START

def _build_adj(n, orientation):
    """Build adjacency matrix from orientation list.
    adj[i][j] = 1 if i -> j, else 0.
    """
    adj = np.zeros((n, n), dtype=np.uint8)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            bit = orientation[idx]
            idx += 1
            if bit == 0:
                adj[i, j] = 1
            else:
                adj[j, i] = 1
    return adj

def _adj_to_orientation(n, adj):
    """Convert adjacency matrix back to orientation list."""
    orientation = []
    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 1:
                orientation.append(0)
            else:
                orientation.append(1)
    return orientation

def _paleyn_tournament(p):
    """Paley tournament on p vertices: i -> j iff (j-i) mod p is a nonzero quadratic residue."""
    squares = set()
    for x in range(1, p):
        squares.add(x * x % p)
    
    orientation = []
    for i in range(p):
        for j in range(i + 1, p):
            diff = (j - i) % p
            if diff in squares:
                orientation.append(0)  # i -> j
            else:
                orientation.append(1)  # j -> i
    return p, orientation

def _is_valid_exhaustive(n, adj):
    """
    Exhaustively check that every 4-set has a common dominator.
    A vertex v dominates a 4-set {a,b,c,d} if v->a, v->b, v->c, v->d.
    Using in-neighborhood bitsets for speed.
    Only feasible for n up to ~45-50 within time limits.
    Returns True if valid, False otherwise.
    """
    # Build in-neighborhood bitsets: in_bits[x] has bit v set iff v -> x
    in_bits = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                in_bits[j] |= (1 << i)
    
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        dominators = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
        if dominators == 0:
            return False
    return True

def _is_valid_sampled(n, adj, num_samples=5000, seed=42):
    """Check validity by sampling random 4-sets."""
    rng = np.random.default_rng(seed)
    
    # Precompute in-neighborhoods as sets
    in_neigh = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                in_neigh[j].add(i)
    
    for _ in range(num_samples):
        S = rng.choice(n, size=4, replace=False)
        a, b, c, d = S
        dominators = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
        if len(dominators) == 0:
            return False
    return True

def _count_violations_exhaustive(n, adj):
    """Count number of 4-sets that lack a common dominator. For small n."""
    in_bits = [0] * n
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                in_bits[j] |= (1 << i)
    
    violations = 0
    for quad in combinations(range(n), 4):
        a, b, c, d = quad
        dominators = in_bits[a] & in_bits[b] & in_bits[c] & in_bits[d]
        if dominators == 0:
            violations += 1
    return violations

def _prune_tournament(n, adj, deadline):
    """
    Try to remove vertices one at a time while preserving the 4-dominator property.
    A vertex v can be removed if every 4-set in the remaining graph still has a dominator.
    But we must also ensure 4-sets that included v are still dominated by some other vertex.
    
    Actually, the property is: every 4-set of vertices in the tournament has a common dominator.
    If we remove vertex v, we need:
    1. Every 4-set not containing v is still dominated (unchanged).
    2. Every 4-set that contained v (i.e., {v, a, b, c}) must now be dominated by some vertex in the remaining set.
       But wait - {v, a, b, c} is a 4-set that included v. After removing v, this 4-set no longer exists in the graph.
       So we only need to check 4-sets in the remaining graph.
    
    Actually, re-reading the problem: "Every set of 4 vertices must have a vertex with arcs to all four."
    The "vertex with arcs to all four" can be any vertex in the tournament, including one of the four.
    So if we remove v, we just need every 4-subset of the remaining vertices to have a dominator in the remaining vertices.
    
    So the pruning criterion is: after removing v, check if the remaining tournament is still valid.
    """
    current_n = n
    current_adj = adj.copy()
    removed = set()
    
    # Try to remove vertices, starting from those that seem least critical
    # Heuristic: try removing vertices with low out-degree or in-degree first
    # Actually, let's try all vertices in order and see which can be removed
    
    while current_n > 5:
        if time.time() > deadline:
            break
        
        # Try to find a vertex that can be removed
        found_removal = False
        for v in range(current_n):
            if time.time() > deadline:
                break
            if v in removed:
                continue
            
            # Check if removing v preserves validity
            # Build the remaining adjacency
            remaining = [u for u in range(current_n) if u not in removed and u != v]
            remaining_set = set(remaining)
            
            if len(remaining) < 4:
                # Too few vertices, can't check 4-sets
                found_removal = True
                removed.add(v)
                current_n -= 1
                break
            
            # Check all 4-sets in remaining
            valid = True
            # Build in-neighborhoods for remaining vertices
            in_bits_rem = {}
            for u in remaining:
                in_bits_rem[u] = 0
                for w in remaining:
                    if u != w and current_adj[u, w] == 1:
                        in_bits_rem[u] |= (1 << w)
            
            for quad in combinations(remaining, 4):
                a, b, c, d = quad
                dominators = in_bits_rem[a] & in_bits_rem[b] & in_bits_rem[c] & in_bits_rem[d]
                if dominators == 0:
                    valid = False
                    break
            
            if valid:
                removed.add(v)
                current_n -= 1
                found_removal = True
                break
        
        if not found_removal:
            break
    
    # Build the final orientation for the remaining vertices
    remaining = [u for u in range(n) if u not in removed]
    remaining_set = set(remaining)
    final_n = len(remaining)
    
    if final_n < 5:
        return n, _adj_to_orientation(n, adj)
    
    # Remap vertices
    remap = {old: new for new, old in enumerate(remaining)}
    final_adj = np.zeros((final_n, final_n), dtype=np.uint8)
    for old_i in remaining:
        for old_j in remaining:
            if old_i != old_j:
                final_adj[remap[old_i], remap[old_j]] = adj[old_i, old_j]
    
    orientation = _adj_to_orientation(final_n, final_adj)
    return final_n, orientation

def _local_search_refine(n, orientation, deadline, max_iterations=5000):
    """
    Local search: try flipping edges to reduce violations, then try pruning.
    Returns (n, orientation, valid).
    """
    adj = _build_adj(n, orientation)
    
    # Precompute in-neighborhoods as sets
    in_neigh = [set() for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j and adj[i, j] == 1:
                in_neigh[j].add(i)
    
    def count_violations_sampled(num_samples=2000, seed=123):
        """Count how many sampled 4-sets are violated."""
        rng = np.random.default_rng(seed)
        violations = 0
        for _ in range(num_samples):
            S = rng.choice(n, size=4, replace=False)
            a, b, c, d = S
            dominators = in_neigh[a] & in_neigh[b] & in_neigh[c] & in_neigh[d]
            if len(dominators) == 0:
                violations += 1
        return violations
    
    # Check if already valid
    if n <= 45:
        if _is_valid_exhaustive(n, adj):
            return n, orientation, True
    else:
        if _is_valid_sampled(n, adj, num_samples=10000, seed=42):
            return n, orientation, True
    
    current_violations = count_violations_sampled()
    best_orientation = orientation[:]
    best_violations = current_violations
    
    if best_violations == 0:
        return n, best_orientation, True
    
    num_pairs = n * (n - 1) // 2
    rng = np.random.default_rng(456)
    
    for iteration in range(max_iterations):
        if time.time() > deadline:
            break
        
        # Pick a random edge to flip
        pair_idx = rng.integers(0, num_pairs)
        # Convert linear index to (i, j)
        i = 0
        while i < n - 1:
            start = i * (2 * n - i - 1) // 2
            end = (i + 1) * (2 * n - i - 2) // 2
            if start <= pair_idx < end:
                break
            i += 1
        j = i + 1 + (pair_idx - i * (2 * n - i - 1) // 2)
        
        # Flip edge (i, j)
        old_dir = adj[i, j]
        adj[i, j] = 1 - old_dir
        adj[j, i] = old_dir
        
        # Update in_neigh
        if old_dir == 0:  # was i -> j, now j -> i
            in_neigh[j].discard(i)
            in_neigh[i].add(j)
        else:  # was j -> i, now i -> j
            in_neigh[i].discard(j)
            in_neigh[j].add(i)
        
        new_violations = count_violations_sampled()
        
        if new_violations < best_violations:
            best_violations = new_violations
            current_violations = new_violations
            best_orientation = _adj_to_orientation(n, adj)
        else:
            # Revert
            adj[i, j] = old_dir
            adj[j, i] = 1 - old_dir
            if old_dir == 0:
                in_neigh[j].add(i)
                in_neigh[i].discard(j)
            else:
                in_neigh[i].add(j)
                in_neigh[j].discard(i)
    
    # Final check on best
    best_adj = _build_adj(n, best_orientation)
    if n <= 45:
        valid = _is_valid_exhaustive(n, best_adj)
    else:
        valid = _is_valid_sampled(n, best_adj, num_samples=10000, seed=42)
    
    return n, best_orientation, valid

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    best_n = 91
    best_orientation = None
    
    # Strategy: 
    # 1. Generate Paley tournaments for primes p ≡ 3 (mod 4) in range [31, 90]
    # 2. For each, check if valid (exhaustive for small, sampled for large)
    # 3. If valid, try to prune to reduce n
    # 4. Track the best (smallest n) valid tournament
    
    # Generate valid Paley primes (p ≡ 3 mod 4)
    valid_paleys = []
    for p in range(31, 91):
        if p >= 2:
            is_prime = True
            for x in range(2, int(p**0.5) + 1):
                if p % x == 0:
                    is_prime = False
                    break
            if is_prime and p % 4 == 3:
                valid_paleys.append(p)
    
    # Try each Paley tournament from smallest to largest
    for p in valid_paleys:
        if time.time() > deadline - 10:
            break
        
        n, orientation = _paleyn_tournament(p)
        adj = _build_adj(n, orientation)
        
        # Check validity
        if n <= 45:
            valid = _is_valid_exhaustive(n, adj)
        else:
            # Use sampled check with more samples for confidence
            valid = _is_valid_sampled(n, adj, num_samples=20000, seed=42)
        
        if not valid:
            continue
        
        # Valid! Try to prune
        pruned_n, pruned_orientation = _prune_tournament(n, adj, deadline - 5)
        
        # Verify pruned tournament is still valid
        if pruned_n >= 5:
            pruned_adj = _build_adj(pruned_n, pruned_orientation)
            if pruned_n <= 45:
                pruned_valid = _is_valid_exhaustive(pruned_n, pruned_adj)
            else:
                pruned_valid = _is_valid_sampled(pruned_n, pruned_adj, num_samples=10000, seed=42)
            
            if pruned_valid and pruned_n < best_n:
                best_n = pruned_n
                best_orientation = pruned_orientation
        elif n < best_n:
            best_n = n
            best_orientation = orientation
    
    # If no valid Paley found, fall back to known p=83
    if best_orientation is None:
        n, orientation = _paleyn_tournament(83)
        best_n = n
        best_orientation = orientation
    
    # Final time check
    if time.time() > deadline:
        pass
    
    return {'n': best_n, 'orientation': best_orientation}

# EVOLVE-BLOCK-END
