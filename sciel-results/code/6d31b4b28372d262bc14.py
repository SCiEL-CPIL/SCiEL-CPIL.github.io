import time
import random
import numpy as np

def _build_seed_coloring():
    """Build the seed 32-vertex colouring from the current program."""
    def mul(a, b):
        r = 0
        for i in range(4):
            if b >> i & 1:
                r ^= a << i
        for i in range(7, 3, -1):
            if r >> i & 1:
                r ^= 0b10011 << (i - 4)
        return r
    log, x = {}, 1
    for e in range(15):
        log[x] = e
        x = mul(x, 2)
    colors = []
    for i in range(32):
        for j in range(i + 1, 32):
            colors.append(log[(i % 16) ^ (j % 16)] % 3 if i // 16 == j // 16 else 3)
    return colors

def _matrices_to_colors(mats, n):
    """Convert 4 adjacency matrices to edge color list in lexicographic order."""
    colors = []
    for i in range(n):
        for j in range(i + 1, n):
            for c in range(4):
                if mats[c][i, j] == 1:
                    colors.append(c)
                    break
    return colors

def _colors_to_matrices(colors, n):
    """Convert edge color list to 4 adjacency matrices."""
    mats = [np.zeros((n, n), dtype=np.int8) for _ in range(4)]
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            c = colors[idx]
            mats[c][i, j] = 1
            mats[c][j, i] = 1
            idx += 1
    return mats

def _count_mono_triangles(mats, n):
    """Count total monochromatic triangles using matrix multiplication."""
    total = 0
    for c in range(4):
        # Number of triangles in color c = trace(A^3) / 6
        A = mats[c].astype(np.int32)
        total += np.trace(A @ A @ A) // 6
    return total

def _extend_vertex_greedy(mats, n, deadline, rng):
    """
    Try to add one vertex (vertex n) to the existing n-vertex coloring.
    For the new vertex, we need to assign a color to each edge (n, i) for i in 0..n-1
    such that no monochromatic triangle is formed.
    
    A triangle (i, j, n) is monochromatic with color c iff:
      mats[c][i, n] == 1 AND mats[c][j, n] == 1 AND mats[c][i, j] == 1
    
    So for each color c, the set S_c = {i : mats[c][i,n]==1} must be an independent set
    in the subgraph defined by mats[c].
    
    We use local search on the assignment vector.
    """
    if time.time() > deadline:
        return None
    
    # For each color c, precompute the adjacency: for vertex i, which vertices j have mats[c][i,j]==1
    # We need: for each c, the set of i assigned color c must have no edges among them in mats[c]
    
    # Local search on assignment vector of length n
    # Cost = number of monochromatic triangles involving the new vertex
    # = sum over c of (number of edges in mats[c] among vertices assigned color c to new vertex)
    
    def compute_cost(assignment):
        """Count mono triangles involving new vertex."""
        cost = 0
        for c in range(4):
            # Vertices assigned color c to new vertex
            verts = np.where(assignment == c)[0]
            if len(verts) < 2:
                continue
            # Count edges among these vertices in mats[c]
            sub = mats[c][np.ix_(verts, verts)]
            cost += np.sum(sub) // 2  # Each edge counted twice
        return cost
    
    def try_add_vertex():
        # Greedy initialization: for each vertex i, pick the color that minimizes conflicts
        # A vertex i can use color c only if no other vertex j with assignment[j]==c has mats[c][i,j]==1
        # Start with random, then improve
        
        best_assignment = None
        best_cost = float('inf')
        
        # Try multiple random restarts
        for restart in range(30):
            if time.time() > deadline:
                return None
            
            # Random init
            assignment = np.array([rng.randint(0, 3) for _ in range(n)])
            
            # Local search: flip one position at a time
            current_cost = compute_cost(assignment)
            for step in range(2000):
                if time.time() > deadline:
                    return None
                
                if current_cost == 0:
                    best_assignment = assignment.copy()
                    best_cost = 0
                    return best_assignment
                
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_assignment = assignment.copy()
                
                # Pick a random position to flip
                pos = rng.randint(0, n - 1)
                old_c = assignment[pos]
                new_c = (old_c + 1 + rng.randint(0, 2)) % 4
                assignment[pos] = new_c
                
                new_cost = compute_cost(assignment)
                
                # Accept if better or with small probability (simulated annealing)
                if new_cost <= current_cost or rng.random() < 0.1:
                    current_cost = new_cost
                else:
                    assignment[pos] = old_c
        
        return best_assignment if best_cost == 0 else None
    
    result = try_add_vertex()
    if result is None:
        return None
    
    # Build new matrices
    new_n = n + 1
    new_mats = [np.zeros((new_n, new_n), dtype=np.int8) for _ in range(4)]
    for c in range(4):
        new_mats[c][:n, :n] = mats[c]
        new_mats[c][:n, new_n-1] = result
        new_mats[c][new_n-1, :n] = result
    return new_mats

def _verify_no_mono_triangle(mats, n):
    """Verify no monochromatic triangle exists."""
    for c in range(4):
        A = mats[c].astype(np.int32)
        t = np.trace(A @ A @ A)
        if t != 0:
            return False
    return True

def construct():
    """
    Construct a 4-colouring of K_n with no monochromatic triangles.
    Maximize n. Returns {'n': n, 'colors': [...]} where colors has one entry
    for each pair (i,j), i<j, in lexicographic order.
    """
    start_time = time.time()
    deadline = start_time + 135  # Safety margin
    
    rng = random.Random(42)
    
    # Start with the seed 32-vertex colouring
    seed_colors = _build_seed_coloring()
    n0 = 32
    mats = _colors_to_matrices(seed_colors, n0)
    
    # Verify seed
    if not _verify_no_mono_triangle(mats, n0):
        return {'n': 1, 'colors': [0]}
    
    best_n = n0
    best_mats = [m.copy() for m in mats]
    
    # Try to extend vertex by vertex
    current_n = n0
    current_mats = mats
    
    while time.time() < deadline:
        new_mats = _extend_vertex_greedy(current_mats, current_n, deadline, rng)
        if new_mats is None:
            break
        current_mats = new_mats
        current_n += 1
        if current_n > best_n:
            best_n = current_n
            best_mats = [m.copy() for m in current_mats]
    
    # If we have time, try different random seeds for the local search
    if time.time() < deadline - 10:
        for seed in range(5):
            if time.time() > deadline:
                break
            rng2 = random.Random(1000 + seed)
            # Rebuild from seed with different perturbation
            pert_colors = seed_colors[:]
            for _ in range(rng2.randint(1, 3)):
                pos = rng2.randint(0, len(pert_colors) - 1)
                pert_colors[pos] = rng2.randint(0, 3)
            
            p_mats = _colors_to_matrices(pert_colors, n0)
            if not _verify_no_mono_triangle(p_mats, n0):
                continue
            
            p_n = n0
            p_mats_cur = p_mats
            while time.time() < deadline:
                new_m = _extend_vertex_greedy(p_mats_cur, p_n, deadline, rng2)
                if new_m is None:
                    break
                p_mats_cur = new_m
                p_n += 1
                if p_n > best_n:
                    best_n = p_n
                    best_mats = [m.copy() for m in p_mats_cur]
    
    # Convert best result to edge color list
    final_colors = _matrices_to_colors(best_mats, best_n)
    
    # Final verification
    if not _verify_no_mono_triangle(best_mats, best_n):
        # Fallback to seed
        return {'n': n0, 'colors': seed_colors}
    
    return {'n': best_n, 'colors': final_colors}
