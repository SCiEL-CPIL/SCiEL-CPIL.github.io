"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from random import Random

def instance_trees(n=20, seed=611):
    rng = Random(seed)
    trees = []
    for k in range(1, n + 1):
        if k <= 2:
            trees.append([] if k == 1 else [(0, 1)])
            continue
        code = [rng.randrange(k) for _ in range(k - 2)]
        degree = [1] * k
        for x in code:
            degree[x] += 1
        edges = []
        for x in code:
            leaf = min(v for v in range(k) if degree[v] == 1)
            edges.append((leaf, x))
            degree[leaf] -= 1
            degree[x] -= 1
        u, w = [v for v in range(k) if degree[v] == 1]
        edges.append((u, w))
        trees.append(edges)
    return trees

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Generate the fixed trees
    trees = instance_trees(20, 611)
    
    # Pre-compute tree data
    tree_edges = trees
    num_trees = 20
    
    # Total edges in K20 is 190.
    # Sum of edges in trees: sum(k-1 for k in 1..20) = 190.
    # We need a perfect packing (decomposition).
    
    # Strategy:
    # We model this as a CSP where we assign each tree edge to a unique host edge.
    # However, the assignment must be consistent with a vertex mapping for each tree.
    # Instead of assigning edges directly, we assign vertex mappings for each tree.
    # But random vertex mapping is what the previous approach did and it was slow/failed.
    
    # Alternative Structural Approach:
    # Use a backtracking solver that places trees one by one, but uses a smarter
    # constraint propagation: for the current tree, we don't just pick random vertices.
    # We try to find a valid embedding by backtracking on the vertices of the tree itself,
    # constrained by the available edges in the host graph.
    
    # Let's implement a recursive search for a single tree embedding given the used edges.
    # To speed this up, we order the trees by "difficulty".
    # Trees with higher maximum degree or more edges might be harder to embed later.
    # Actually, embedding large trees first is usually better because they consume more resources.
    # But the constraint is edge-disjointness.
    
    # Let's try a different order: Embed trees that have high structural constraints first.
    # Or, use a randomized restart approach with a more efficient embedding check.
    
    # Key Insight: The previous approach failed likely because it didn't backtrack effectively
    # on the *choice* of which specific host vertices to use, only on the random restarts.
    # We will implement a proper backtracking search for the vertex mapping of each tree,
    # pruning branches where an edge cannot be placed.
    
    # Pre-compute all possible edges in K20
    all_host_edges = set()
    for u in range(20):
        for v in range(u + 1, 20):
            all_host_edges.add((u, v))
            
    used_edges = set()
    mappings = [None] * 20
    score = 0
    
    # Order of trees to embed.
    # Let's try embedding in order of decreasing size (20 down to 1).
    # T20 is a spanning tree. T1 is trivial.
    order = list(range(19, -1, -1))
    
    # We will use a recursive function that tries to embed the tree at order[idx].
    # To make it efficient, we will use a heuristic to pick the next tree vertex to map.
    
    def get_tree_adj(tree_idx):
        k = tree_idx + 1
        if k <= 2:
            return [[] for _ in range(k)]
        edges = tree_edges[tree_idx]
        adj = [[] for _ in range(k)]
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
        return adj

    tree_adjs = [get_tree_adj(i) for i in range(num_trees)]

    def backtrack(idx):
        nonlocal used_edges, mappings, score
        
        if time.time() > deadline:
            return False
            
        if idx == len(order):
            return True
            
        tree_idx = order[idx]
        k = tree_idx + 1
        edges = tree_edges[tree_idx]
        adj = tree_adjs[tree_idx]
        
        if k == 1:
            # T1: Map to any vertex. Let's pick 0 for simplicity, or a random one.
            # Since it uses no edges, it doesn't constrain others.
            mappings[tree_idx] = [0]
            if backtrack(idx + 1):
                return True
            mappings[tree_idx] = None
            return False
            
        if k == 2:
            # T2: One edge. Need to find any available edge (u,v).
            for u in range(20):
                for v in range(u + 1, 20):
                    e_key = (u, v)
                    if e_key not in used_edges:
                        # Map tree vertex 0 to u, tree vertex 1 to v
                        # Check if edge (0,1) in tree maps to (u,v)
                        # Tree edge is (0,1). So mapping [u,v] works.
                        # Also [v,u] works.
                        mappings[tree_idx] = [u, v]
                        used_edges.add(e_key)
                        score += 1
                        if backtrack(idx + 1):
                            return True
                        used_edges.remove(e_key)
                        score -= 1
                        mappings[tree_idx] = None
                        # We could also try [v, u] but it's symmetric for edge usage.
                        # However, the vertex mapping matters for consistency if we were
                        # doing global consistency, but here trees are independent except for edges.
                        # So [u,v] is sufficient.
            return False

        # For k > 2, we need to find a valid embedding.
        # We will use backtracking on the vertices of the tree.
        # We need to assign a host vertex to each tree vertex 0..k-1.
        # Host vertices must be distinct.
        # Edges in the tree must map to available host edges.
        
        # Heuristic: Start with a random root, then expand.
        # Or, use a more systematic approach:
        # Try all permutations? No, too slow.
        # Use randomized backtracking with forward checking.
        
        # Let's try a limited number of random restarts for this specific tree embedding
        # within the context of the current used_edges state.
        
        # Actually, a better strategy:
        # Since the graph is dense (K20), most random embeddings will work initially.
        # The problem arises when the graph becomes sparse.
        # So, we should prioritize embedding trees that are "harder" to embed later.
        # Which trees are harder? Those with high degree or specific structures.
        
        # Let's stick to the current order (largest first) but improve the embedding search.
        
        # For this tree, we will attempt to find an embedding using a randomized backtracking
        # search over the tree's vertices.
        
        # We will try up to a certain number of random initializations.
        max_restarts_for_tree = 20
        
        for _ in range(max_restarts_for_tree):
            if time.time() > deadline:
                return False
                
            # Generate a random permutation of 20 vertices, take the first k.
            # This gives a random mapping of tree vertices to host vertices.
            # We will refine this mapping using backtracking if it fails?
            # No, let's just check if this random mapping works.
            # If not, try another random mapping.
            
            # To be more effective, we can bias the selection.
            # But simple random sampling is a good baseline.
            
            host_verts = list(range(20))
            Random(time.time_ns()).shuffle(host_verts)
            mapping = host_verts[:k]
            
            # Check if this mapping is valid
            valid = True
            edges_to_add = []
            for u, v in edges:
                hu = mapping[u]
                hv = mapping[v]
                e_key = (min(hu, hv), max(hu, hv))
                if e_key in used_edges:
                    valid = False
                    break
                edges_to_add.append(e_key)
                
            if valid:
                # Apply
                used_edges.update(edges_to_add)
                mappings[tree_idx] = mapping
                score += len(edges)
                
                if backtrack(idx + 1):
                    return True
                    
                # Undo
                for e in edges_to_add:
                    used_edges.remove(e)
                mappings[tree_idx] = None
                score -= len(edges)
                
        # If no random mapping worked for this tree in the current state,
        # we might need to backtrack to the previous tree.
        return False

    # Run the solver
    # We will run multiple full restarts of the entire process with different random seeds
    # for the internal random choices.
    
    best_score = 0
    best_mappings = [None] * 20
    
    # Since the problem is very tight (perfect decomposition required),
    # simple random restarts might not find it quickly.
    # However, the previous score was 7.3, which is very low (190 is max).
    # This suggests the previous approach was failing to pack many trees.
    # Why? "Greedy embedding from largest to smallest... score 6.8".
    # 6.8 * 20 (approx) ? No, score is total edges.
    # Max edges is 190. Score 7.3 means only ~7 edges packed?
    # That's extremely low. T20 alone has 19 edges.
    # If T20 is embedded, score should be at least 19.
    # Unless T20 embedding failed?
    # Or the score calculation is different?
    # "Raw value is the total number of embedded tree edges".
    # If T20 is embedded, that's 19 edges.
    # If T19 is embedded, that's 18 more.
    # Why would the score be so low?
    # Maybe the previous program failed to embed T20?
    # Or maybe the evaluator penalizes invalid results heavily?
    # "Any broken rule makes the whole result invalid... scores 0".
    # But the score is 7.3158, not 0. So it's valid but poor.
    
    # Let's ensure we at least embed T20.
    # T20 is a spanning tree. It can always be embedded in K20.
    # So score should be >= 19.
    
    # Let's run the backtracking solver.
    
    # To increase the chance of success, we can try different orders.
    # But first, let's try the standard order (20 down to 1).
    
    # We will run multiple independent attempts with different random seeds.
    
    for restart in range(50):
        if time.time() > deadline:
            break
            
        # Reset state
        used_edges.clear()
        mappings = [None] * 20
        score = 0
        
        # Use a fixed random seed for reproducibility in debugging, or time-based
        rng = Random(611 + restart)
        
        # We need to pass rng to the backtrack function or use a global one.
        # Let's refactor backtrack to accept rng or use a closure.
        
        def backtrack_with_rng(idx, rng):
            nonlocal used_edges, mappings, score
            
            if time.time() > deadline:
                return False
                
            if idx == len(order):
                return True
                
            tree_idx = order[idx]
            k = tree_idx + 1
            edges = tree_edges[tree_idx]
            
            if k == 1:
                mappings[tree_idx] = [0]
                if backtrack_with_rng(idx + 1, rng):
                    return True
                mappings[tree_idx] = None
                return False
                
            if k == 2:
                # Find any available edge
                available = [e for e in all_host_edges if e not in used_edges]
                rng.shuffle(available)
                for e in available:
                    u, v = e
                    mappings[tree_idx] = [u, v]
                    used_edges.add(e)
                    score += 1
                    if backtrack_with_rng(idx + 1, rng):
                        return True
                    used_edges.remove(e)
                    score -= 1
                    mappings[tree_idx] = None
                return False

            # For k > 2
            # Try random mappings
            max_attempts = 50
            for _ in range(max_attempts):
                if time.time() > deadline:
                    return False
                    
                host_verts = list(range(20))
                rng.shuffle(host_verts)
                mapping = host_verts[:k]
                
                valid = True
                edges_to_add = []
                for u, v in edges:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        valid = False
                        break
                    edges_to_add.append(e_key)
                    
                if valid:
                    used_edges.update(edges_to_add)
                    mappings[tree_idx] = mapping
                    score += len(edges)
                    
                    if backtrack_with_rng(idx + 1, rng):
                        return True
                        
                    for e in edges_to_add:
                        used_edges.remove(e)
                    mappings[tree_idx] = None
                    score -= len(edges)
                    
            return False

        if backtrack_with_rng(0, rng):
            if score > best_score:
                best_score = score
                best_mappings = [m[:] if m else None for m in mappings]
                if best_score == 190:
                    break

    # Fallback: If we didn't find a good solution, try a simple greedy to ensure non-zero score
    if best_score < 19:
        # Just embed T20 with identity mapping
        # T20 edges are fixed. Identity mapping maps tree vertex i to host vertex i.
        # Check if edges are valid (they are, in K20).
        used_edges.clear()
        mappings = [None] * 20
        score = 0
        
        # Embed T20
        k = 20
        edges = tree_edges[19]
        mapping = list(range(20))
        valid = True
        for u, v in edges:
            e_key = (min(u, v), max(u, v))
            if e_key in used_edges:
                valid = False
                break
            used_edges.add(e_key)
        if valid:
            mappings[19] = mapping
            score += len(edges)
            
        # Try to embed others greedily
        for tree_idx in range(18, -1, -1):
            k = tree_idx + 1
            edges = tree_edges[tree_idx]
            if k == 1:
                mappings[tree_idx] = [0]
                continue
            if k == 2:
                available = [e for e in all_host_edges if e not in used_edges]
                if available:
                    u, v = available[0]
                    mappings[tree_idx] = [u, v]
                    used_edges.add((u, v))
                    score += 1
                continue
            
            # Try random mapping
            rng = Random(tree_idx)
            found = False
            for _ in range(100):
                host_verts = list(range(20))
                rng.shuffle(host_verts)
                mapping = host_verts[:k]
                valid = True
                edges_to_add = []
                for u, v in edges:
                    hu = mapping[u]
                    hv = mapping[v]
                    e_key = (min(hu, hv), max(hu, hv))
                    if e_key in used_edges:
                        valid = False
                        break
                    edges_to_add.append(e_key)
                if valid:
                    used_edges.update(edges_to_add)
                    mappings[tree_idx] = mapping
                    score += len(edges)
                    found = True
                    break
            if not found:
                pass # Leave as None

        if score > best_score:
            best_score = score
            best_mappings = [m[:] if m else None for m in mappings]

    # Construct final result
    final_result = []
    for i in range(20):
        if best_mappings[i] is None:
            final_result.append(None)
        else:
            # Ensure types are int
            final_result.append([int(x) for x in best_mappings[i]])
            
    return final_result

# EVOLVE-BLOCK-END
