import numpy as np
import random
import itertools

def construct(q, n, radius):
    # Parameters for this specific task instance
    # The evaluator passes q, n, radius. 
    # Based on the task description, q=3, n=6, radius=1.
    # We assume the evaluator calls with these values.
    
    if q != 3 or n != 6 or radius != 1:
        # Fallback to a general simple construction if parameters differ, 
        # though the task contract implies this specific instance.
        # For safety, if it's not the specific case, use a greedy approach generally.
        pass

    total_vertices = q ** n
    vertices = list(itertools.product(range(q), repeat=n))
    vertex_to_idx = {v: i for i, v in enumerate(vertices)}
    
    # Pre-compute neighbors (distance 1) for each vertex
    # neighbors[i] is a list of indices of vertices at Hamming distance 1 from vertices[i]
    neighbors = []
    for v in vertices:
        v_list = list(v)
        nbrs = []
        for pos in range(n):
            old_val = v_list[pos]
            for new_val in range(q):
                if new_val != old_val:
                    v_list[pos] = new_val
                    nbrs.append(vertex_to_idx[tuple(v_list)])
            v_list[pos] = old_val
        # Include self? 
        # Dominating set definition: v is dominated if v in S or v adjacent to S.
        # So S covers S and neighbors(S).
        # We will treat "coverage" as self + neighbors.
        neighbors.append(nbrs)
        
    # For fast coverage checks, let's also store the "ball" of radius 1 (self + neighbors)
    balls = []
    for i in range(total_vertices):
        ball = set(neighbors[i])
        ball.add(i)
        balls.append(ball)

    def is_dominating(set_indices):
        """Check if the set of indices dominates the whole space."""
        if not set_indices:
            return total_vertices == 0
        covered = set()
        for idx in set_indices:
            covered.update(balls[idx])
        return len(covered) == total_vertices

    def greedy_dominating_set(random_seed=None):
        if random_seed is not None:
            random.seed(random_seed)
            
        undominated = set(range(total_vertices))
        dominating_set = []
        
        while undominated:
            # Find vertex that covers most undominated vertices
            best_vertex = None
            max_covered = -1
            # Optimization: only consider vertices that are in undominated or adjacent to undominated?
            # Actually, any vertex can be picked. But picking one that is already dominated is wasteful 
            # unless it covers many new ones. 
            # However, usually greedy picks from the pool of candidates that cover new nodes.
            
            # To speed up, we can just iterate through all vertices, but that's 729 * 729 per step?
            # No, we iterate through candidates. 
            # A better heuristic: Pick a random undominated vertex u. 
            # Candidates are u and its neighbors. Pick the one in candidates that covers the most undominated.
            # This is the "Local Search" greedy variant.
            
            if not undominated:
                break
                
            # Pick a random target to focus on
            target = random.choice(list(undominated))
            candidates = set(balls[target]) # target and its neighbors
            
            # Score candidates by how many undominated vertices they cover
            best_score = -1
            best_cand = None
            for c in candidates:
                covered_count = len(undominated.intersection(balls[c]))
                # Add small random noise to break ties
                score = covered_count + random.random() * 0.1
                if score > best_score:
                    best_score = score
                    best_cand = c
            
            dominating_set.append(best_cand)
            undominated -= balls[best_cand]
            
        return dominating_set

    def minimize_by_deletion(dominating_set):
        """Remove redundant codewords."""
        # Convert to set for fast lookup
        S = set(dominating_set)
        
        # Count how many codewords cover each vertex
        # count[v] = number of c in S such that v in balls[c]
        counts = np.zeros(total_vertices, dtype=int)
        for c in S:
            for v in balls[c]:
                counts[v] += 1
        
        changed = True
        while changed:
            changed = False
            # Iterate in random order to avoid bias
            for c in random.sample(list(S), len(S)):
                # Check if c is critical
                # c is critical if there exists v in balls[c] such that counts[v] == 1
                # If all v in balls[c] have counts[v] > 1, we can remove c.
                critical = False
                for v in balls[c]:
                    if counts[v] == 1:
                        critical = True
                        break
                
                if not critical:
                    # Remove c
                    S.remove(c)
                    for v in balls[c]:
                        counts[v] -= 1
                    changed = True
        
        return list(S)

    best_size = float('inf')
    best_set = None
    
    # Run multiple randomizations to find a smaller set
    # 20 iterations should be fast enough (729 vertices is small)
    for i in range(30):
        ds = greedy_dominating_set(random_seed=i)
        minimized = minimize_by_deletion(ds)
        if len(minimized) < best_size:
            best_size = len(minimized)
            best_set = minimized
            
        # Early stopping if we hit the known lower bound or optimal
        if best_size <= 73:
            break

    if best_set is None:
        # Fallback if something went wrong (should not happen)
        best_set = greedy_dominating_set(0)
        best_set = minimize_by_deletion(best_set)

    # Convert indices back to codewords
    return [vertices[idx] for idx in best_set]
