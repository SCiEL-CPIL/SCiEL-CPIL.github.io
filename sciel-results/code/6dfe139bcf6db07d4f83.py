import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 115
    
    n = 20
    total_edges = n * (n - 1) // 2
    
    # Precompute edge indices for vectorization
    # edge_map[i] = index of edge i in the flattened matrix
    # We represent the graph as an adjacency matrix A (n x n)
    
    # Helper: get adjacency matrix from edge list (not used in main loop, just for output)
    def edges_to_matrix(edges):
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in edges:
            A[u, v] = 1
            A[v, u] = 1
        return A

    # Helper: get edge list from matrix
    def matrix_to_edges(A):
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if A[i, j]:
                    edges.append([i, j])
        return edges

    # Vectorized Fitness Function
    # Returns: (edge_count, violation_count)
    # violation_count is the number of triples (i,j,k) with >= 3 common neighbors
    def evaluate(A):
        # A is n x n int8 matrix
        # Common neighbors of i and j is (A @ A)[i, j]
        # We need to check if for any triple (i, j, k), the number of common neighbors is >= 3.
        # Actually, the condition is: no three vertices have three common neighbours.
        # This is equivalent to: For any set of 3 vertices S = {i, j, k}, |N(i) ∩ N(j) ∩ N(k)| < 3.
        
        # Let C = A @ A. C[i, j] is the number of common neighbors of i and j.
        # This doesn't directly give the triple intersection.
        
        # A faster way: 
        # For each vertex v, let N(v) be its neighbors.
        # A K3,3 exists if there are 3 vertices i,j,k such that at least 3 vertices in N(i)∩N(j)∩N(k).
        
        # Vectorized approach for n=20:
        # We can iterate over pairs (i, j) and compute common neighbors.
        # But we need triples.
        
        # Let's use the property: 
        # If we compute M = A @ A @ A, M[i, j] is the number of walks of length 3 from i to j.
        # This is not quite the intersection size.
        
        # The most efficient vectorized way for small n:
        # For each vertex v, consider the subgraph induced by its neighbors? No.
        
        # Let's stick to a semi-vectorized approach that is fast enough for n=20.
        # For each pair (i, j), common neighbors are mask_i & mask_j.
        # We need to check if any triple (i, j, k) has |mask_i & mask_j & mask_k| >= 3.
        
        # Since n=20, there are C(20,3) = 1140 triples.
        # We can compute the intersection size for all triples at once using numpy?
        
        # Let A be the adjacency matrix.
        # For a triple (i, j, k), the common neighbors are rows i, j, k of A, ANDed together.
        # Sum of that AND is the count.
        
        # We can represent rows of A as bitmasks (integers) and use bitwise ops? 
        # Numpy bitwise ops on int64 are fast.
        
        # Convert A rows to int64 bitmasks
        # Bit position v corresponds to vertex v.
        
        # However, doing this in pure Python loop over 1140 triples is very fast (< 1ms).
        # Let's try to vectorize the triple check.
        
        # Alternative: 
        # Count violations by checking all triples.
        # To make it fast in Python, we can precompute bitmasks.
        
        # But the directive asks for "fast vectorized common-neighbor counts".
        # Let's try to do it with numpy matrix operations if possible, or just fast Python bit ops.
        
        # Actually, for n=20, Python bit ops on integers are extremely fast.
        # Let's convert A to a list of integers (bitmasks) once.
        
        # This function is called frequently, so we should optimize the input format.
        # But the GA operates on matrices or edge lists. 
        # Let's operate on bitmasks directly in the GA representation!
        # Representation: List of 20 integers, each a 20-bit mask.
        
        # Wait, the function signature takes A (matrix) or we can change the internal representation.
        # The prompt says "fitness is calculated as ... via fast vectorized common-neighbor counts".
        # I will implement the GA using bitmasks for speed, but the final output must be edges.
        
        # Let's define the fitness function on bitmasks.
        pass # Placeholder, will define inside GA loop for access to local variables if needed, 
             # but better to keep it clean.

    def fitness_bitmasks(masks):
        """
        masks: list of 20 integers (bitmasks)
        Returns: (edge_count, violations)
        """
        edge_count = sum(m.bit_count() for m in masks) // 2
        
        # Check for K3,3 violations
        # Triple (i, j, k) has violation if (masks[i] & masks[j] & masks[k]).bit_count() >= 3
        violations = 0
        # Precompute pairwise intersections to speed up?
        # There are 190 pairs.
        # Then for each pair (i, j), we check intersection with all k > j.
        
        # This is O(n^3) bit ops. 20^3 = 8000 ops. Very fast.
        for i in range(n):
            mi = masks[i]
            for j in range(i + 1, n):
                mij = mi & masks[j]
                if mij == 0:
                    continue
                # If mij has < 3 bits, it can't form a violation with any k
                if mij.bit_count() < 3:
                    continue
                for k in range(j + 1, n):
                    if (mij & masks[k]).bit_count() >= 3:
                        violations += 1
                        # Early exit? No, we need total penalty for GA fitness.
                        # But if violations > 0, fitness is bad.
        
        return edge_count, violations

    # GA Setup
    pop_size = 100
    mutation_rate = 0.1
    crossover_rate = 0.8
    
    # Initialize Population
    # Start with a mix of random and structured graphs
    population = []
    
    # 1. Random dense graphs
    for _ in range(pop_size // 2):
        masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < 0.6:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        population.append(masks)
        
    # 2. Structured base (Polarity-like or Grid)
    def create_structured():
        masks = [0] * n
        # Simple grid 4x5
        # Connect neighbors in grid
        for r in range(4):
            for c in range(5):
                u = r * 5 + c
                # Right
                if c < 4:
                    v = r * 5 + c + 1
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                # Down
                if r < 3:
                    v = (r + 1) * 5 + c
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                # Diagonals
                if c < 4 and r < 3:
                    v = (r + 1) * 5 + c + 1
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
                if c > 0 and r < 3:
                    v = (r + 1) * 5 + c - 1
                    masks[u] |= (1 << v)
                    masks[v] |= (1 << u)
        # Add some random extra edges to increase density
        for i in range(n):
            for j in range(i + 1, n):
                if not (masks[i] & (1 << j)) and random.random() < 0.3:
                    masks[i] |= (1 << j)
                    masks[j] |= (1 << i)
        return masks

    for _ in range(pop_size // 2):
        population.append(create_structured())
        
    # Evaluate initial population
    scores = []
    for masks in population:
        ec, viol = fitness_bitmasks(masks)
        # Fitness: Maximize edges, penalize violations heavily
        # If violations > 0, fitness is very low
        # We want to find valid graphs (viol=0) with max edges.
        # Fitness = edge_count if viol == 0 else edge_count - 1000 * viol
        # Or simpler: Lexicographic: (viol==0, edge_count)
        # For GA, a continuous fitness is better.
        # Let's use: fitness = edge_count - 100 * viol
        fit = ec - 100 * viol
        scores.append(fit)
        
    # Track best
    best_masks = None
    best_fit = -1
    best_ec = -1
    
    def update_best(masks, fit, ec, viol):
        nonlocal best_masks, best_fit, best_ec
        if viol == 0:
            if ec > best_ec:
                best_ec = ec
                best_fit = fit
                best_masks = masks[:]
        else:
            # If we haven't found a valid one yet, track best invalid?
            # No, we only care about valid ones for the final answer.
            # But for GA selection, we use the fitness score.
            if fit > best_fit:
                best_fit = fit
                best_ec = ec # Note: this ec is for an invalid graph, but best_ec is only updated if viol==0
                # We need to be careful. best_masks should only be valid if we intend to return it.
                # If no valid graph is found, we must return an empty list or a valid one.
                # The task requires a K3,3-free graph.
                # So we MUST return a valid graph.
                # If GA fails to find a valid one, we fall back to a simple valid construction.
                
                # Let's keep a separate "best_valid" and "best_any"
                pass

    # Let's refine the tracking
    best_valid_masks = None
    best_valid_ec = -1
    
    for i in range(pop_size):
        ec, viol = fitness_bitmasks(population[i])
        if viol == 0:
            if ec > best_valid_ec:
                best_valid_ec = ec
                best_valid_masks = population[i][:]
                
    # GA Loop
    gen = 0
    while time.time() < deadline:
        gen += 1
        
        # Selection: Tournament Selection
        def select():
            # Pick 3 random individuals, return the best
            idx1 = random.randint(0, pop_size - 1)
            idx2 = random.randint(0, pop_size - 1)
            idx3 = random.randint(0, pop_size - 1)
            
            best_idx = idx1
            if scores[idx2] > scores[best_idx]:
                best_idx = idx2
            if scores[idx3] > scores[best_idx]:
                best_idx = idx3
            return population[best_idx]
        
        new_population = []
        
        # Elitism: Keep the top 10%
        sorted_indices = np.argsort(scores)[::-1]
        elite_count = max(1, pop_size // 10)
        for i in range(elite_count):
            idx = sorted_indices[i]
            new_population.append(population[idx][:])
        
        # Generate offspring
        while len(new_population) < pop_size:
            if time.time() > deadline:
                break
                
            p1 = select()
            p2 = select()
            
            # Crossover
            if random.random() < crossover_rate:
                # Uniform Crossover
                child = [0] * n
                for i in range(n):
                    for j in range(i + 1, n):
                        if random.random() < 0.5:
                            if p1[i] & (1 << j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
                        else:
                            if p2[i] & (1 << j):
                                child[i] |= (1 << j)
                                child[j] |= (1 << i)
            else:
                child = p1[:]
            
            # Mutation
            # Randomly flip edges
            # To avoid too many mutations, we flip a small number of edges
            num_mutations = int(n * n * mutation_rate / 2) # Roughly 10-20 edges
            for _ in range(num_mutations):
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v:
                    continue
                if child[u] & (1 << v):
                    child[u] &= ~(1 << v)
                    child[v] &= ~(1 << u)
                else:
                    child[u] |= (1 << v)
                    child[v] |= (1 << u)
            
            new_population.append(child)
            
        # Evaluate new population
        population = new_population
        scores = []
        for masks in population:
            ec, viol = fitness_bitmasks(masks)
            fit = ec - 100 * viol
            scores.append(fit)
            
            if viol == 0:
                if ec > best_valid_ec:
                    best_valid_ec = ec
                    best_valid_masks = masks[:]
        
        # Adaptive Mutation Rate?
        # If we found a valid graph, reduce mutation to refine.
        # If not, increase mutation to explore.
        if best_valid_ec > 0:
            mutation_rate = max(0.01, mutation_rate * 0.95)
        else:
            mutation_rate = min(0.3, mutation_rate * 1.05)
            
        # Check time
        if time.time() > deadline - 5:
            break

    # Fallback: If no valid graph found by GA, use a simple deterministic valid graph
    if best_valid_masks is None:
        # Empty graph is valid
        best_valid_masks = [0] * n
        best_valid_ec = 0
        
        # Try to add edges greedily to the empty graph to get a better valid base
        # This is just a safety net. The GA should find something better.
        masks = [0] * n
        for i in range(n):
            for j in range(i + 1, n):
                # Try adding edge (i, j)
                masks[i] |= (1 << j)
                masks[j] |= (1 << i)
                
                # Check validity
                valid = True
                # Fast check: only check triples involving i or j
                # (Simplified check for fallback, might be slow if many edges)
                # For n=20, full check is fast enough if done sparingly.
                # Let's just do a full check since it's a fallback
                for a in range(n):
                    ma = masks[a]
                    for b in range(a + 1, n):
                        mab = ma & masks[b]
                        for c in range(b + 1, n):
                            if (mab & masks[c]).bit_count() >= 3:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                if not valid:
                    masks[i] &= ~(1 << j)
                    masks[j] &= ~(1 << i)
                    
        ec, viol = fitness_bitmasks(masks)
        if viol == 0 and ec > best_valid_ec:
            best_valid_ec = ec
            best_valid_masks = masks

    # Final Polish: Local Search on the best valid graph found
    if best_valid_masks is not None and time.time() < deadline - 2:
        masks = best_valid_masks[:]
        
        def try_add_edge(m, u, v):
            # Temporarily add
            m[u] |= (1 << v)
            m[v] |= (1 << u)
            
            # Check validity
            valid = True
            # Check triples involving u or v
            for w in range(n):
                if w == u or w == v: continue
                mw = m[w]
                # Triple {u, v, w}
                if (m[u] & m[v] & mw).bit_count() >= 3:
                    valid = False
                    break
                for z in range(w + 1, n):
                    if z == u or z == v: continue
                    mz = m[z]
                    if (m[u] & mw & mz).bit_count() >= 3:
                        valid = False
                        break
                    if (m[v] & mw & mz).bit_count() >= 3:
                        valid = False
                        break
                if not valid: break
            
            if not valid:
                m[u] &= ~(1 << v)
                m[v] &= ~(1 << u)
                return False
            return True
            
        improved = True
        while improved and time.time() < deadline - 2:
            improved = False
            # Try adding all missing edges
            for u in range(n):
                for v in range(u + 1, n):
                    if not (masks[u] & (1 << v)):
                        if try_add_edge(masks, u, v):
                            improved = True
                            best_valid_ec = sum(m.bit_count() for m in masks) // 2
                            
    # Convert best masks to edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if best_valid_masks[i] & (1 << j):
                edges.append([i, j])
                
    return edges
