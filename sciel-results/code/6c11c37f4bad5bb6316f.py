import numpy as np
import random
import itertools

def construct(q, n, radius):
    # Specific task: q=3, n=6, radius=1
    if q != 3 or n != 6 or radius != 1:
        # Fallback
        pass

    total_vertices = q ** n
    # Generate all vertices
    # Use numpy for speed
    # vertices is shape (729, 6)
    # We can generate indices and convert to base-q
    indices = np.arange(total_vertices)
    vertices = np.zeros((total_vertices, n), dtype=int)
    for pos in range(n):
        div = q ** (n - 1 - pos)
        vertices[:, pos] = (indices // div) % q

    # Precompute balls (self + neighbors)
    # Ball of radius 1 for vertex i covers i and its n*(q-1) neighbors.
    # We need to map vertex tuples to indices.
    # Since vertices are generated in order, we can use the index directly if we generate them in a specific way?
    # No, `itertools.product` generates in lexicographic order, which matches base-q expansion if we are careful.
    # Let's stick to a mapping dict for safety, or just use the index if we ensure generation order.
    # The current code uses `itertools.product` and a dict.
    # With 729 vertices, the dict is fine.
    
    vertices_list = [tuple(v) for v in vertices]
    vertex_to_idx = {v: i for i, v in enumerate(vertices_list)}
    
    # Compute balls as lists of indices
    balls = []
    for i in range(total_vertices):
        v = vertices_list[i]
        ball = {i}
        for pos in range(n):
            for new_val in range(q):
                if new_val != v[pos]:
                    new_v = list(v)
                    new_v[pos] = new_val
                    ball.add(vertex_to_idx[tuple(new_v)])
        balls.append(list(ball))
        
    # Convert balls to numpy arrays for faster set operations? 
    # Or just use sets. 729 * 13 elements is small.
    balls_sets = [set(b) for b in balls]

    def is_dominating(set_indices):
        if not set_indices:
            return total_vertices == 0
        covered = set()
        for idx in set_indices:
            covered.update(balls_sets[idx])
        return len(covered) == total_vertices

    def greedy_dominating_set(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        undominated = set(range(total_vertices))
        dominating_set = []
        
        while undominated:
            # Pick a random target
            target = random.choice(list(undominated))
            # Candidates are the ball of the target
            candidates = balls_sets[target]
            
            best_score = -1
            best_cand = None
            for c in candidates:
                # Count how many undominated vertices c covers
                # Using set intersection
                covered_count = len(undominated.intersection(balls_sets[c]))
                score = covered_count + random.random() * 0.01
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            dominating_set.append(best_cand)
            undominated -= balls_sets[best_cand]
            
        return dominating_set

    def local_search(dominating_set, max_iters=50):
        """
        Try to improve the set by:
        1. Removing redundant nodes (deletion).
        2. Swapping a node for another if it allows further deletion.
        """
        S = set(dominating_set)
        
        # Helper to check if S is dominating
        def covers_all(S):
            covered = set()
            for idx in S:
                covered.update(balls_sets[idx])
            return len(covered) == total_vertices

        # First, do pure deletion to get a minimal set
        changed = True
        while changed:
            changed = False
            # Random order
            for c in random.sample(list(S), len(S)):
                # Check if c can be removed
                # Remove c, check if still dominating
                S.remove(c)
                if covers_all(S):
                    changed = True
                else:
                    S.add(c)
        
        # Now try swaps
        for _ in range(max_iters):
            improved = False
            for c in list(S):
                # Try replacing c with some other vertex v
                # We can sample v from the whole space or just neighbors of c?
                # Replacing with a distant vertex is unlikely to help immediately unless it covers a "hole".
                # Let's try all vertices? 729 * |S| ~ 729 * 73 ~ 50k checks.
                # Each check involves removing c, adding v, and checking coverage.
                # Checking coverage is expensive (O(|S| * ball_size)).
                # Optimize: 
                # 1. Remove c. Calculate coverage C1.
                # 2. If C1 == total, we can just remove c (handled above).
                # 3. If C1 < total, let U = uncovered.
                # 4. We need v such that balls[v] covers U.
                #    i.e., U is subset of balls[v].
                #    This means v must be in the intersection of balls[u] for all u in U?
                #    No, v must be such that for all u in U, u is in balls[v].
                #    This is equivalent to v being in the "common ball" of U?
                #    Actually, if U is small, we can just iterate v.
                
                S.remove(c)
                covered = set()
                for idx in S:
                    covered.update(balls_sets[idx])
                uncovered = set(range(total_vertices)) - covered
                
                if not uncovered:
                    # Should have been caught by deletion, but just in case
                    improved = True
                    break
                
                # Find v that covers all uncovered
                # v must be in intersection of balls[u] for u in uncovered?
                # No. v covers u if u in balls[v].
                # So we need v such that uncovered subset of balls[v].
                # This means v is in the intersection of "pre-image" of uncovered?
                # Let's just iterate v. 
                # Optimization: Only v that are in the ball of some uncovered node?
                # If v covers u, then v is in balls[u].
                # So v must be in intersection of balls[u] for all u in uncovered?
                # Yes! If v covers all u in U, then v is in balls[u] for each u in U.
                # So v must be in intersection_{u in U} balls[u].
                
                candidates = set(balls_sets[next(iter(uncovered))])
                for u in list(uncovered)[1:]:
                    candidates &= balls_sets[u]
                    if not candidates:
                        break
                
                if candidates:
                    # Pick a random candidate v
                    v = random.choice(list(candidates))
                    if v in S:
                        continue
                    # Add v
                    S.add(v)
                    # Now S is dominating.
                    # Try to delete more from S (including c's old neighbors?)
                    # Just run deletion again on the new S
                    # But we are in a loop. Let's just mark improved and break to restart deletion?
                    # Or do a quick deletion pass.
                    old_size = len(S)
                    # Quick deletion
                    changed2 = True
                    while changed2:
                        changed2 = False
                        for c2 in random.sample(list(S), len(S)):
                            S.remove(c2)
                            if covers_all(S):
                                changed2 = True
                            else:
                                S.add(c2)
                    if len(S) < old_size:
                        improved = True
                        break
                    else:
                        # Revert if no gain? 
                        # The swap resulted in same size. 
                        # We might want to keep it if it enables future swaps?
                        # For now, revert to keep it simple.
                        S.remove(v)
            
            if not improved:
                break
                
        return list(S)

    best_size = float('inf')
    best_set = None
    
    # Run multiple iterations
    # Time budget: 120s.
    # Each iteration: Greedy (fast) + Local Search (slower).
    # Let's run as many as possible.
    
    import time
    start_time = time.time()
    seed = 0
    while time.time() - start_time < 110:
        ds = greedy_dominating_set(random_seed=seed)
        minimized = local_search(ds)
        if len(minimized) < best_size:
            best_size = len(minimized)
            best_set = minimized
            if best_size <= 73:
                break
        seed += 1

    if best_set is None:
        best_set = greedy_dominating_set(0)
        best_set = local_search(best_set)

    return [vertices_list[idx] for idx in best_set]
