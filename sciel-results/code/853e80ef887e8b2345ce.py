"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import sys
from collections import defaultdict

def construct():
    # Deadline guard: 150s wall-clock limit, use 145s for safety
    start_time = time.time()
    deadline = start_time + 145.0

    # 1. Generate Instance Graphs (copied from verifier logic)
    def instance_graph(n, p, seed):
        rng = random.Random(seed)
        order = list(range(n))
        rng.shuffle(order)
        edges = set()
        for i in range(n - 1):
            u, v = order[i], order[i + 1]
            edges.add(tuple(sorted((u, v))))
        for u in range(n):
            for v in range(u + 1, n):
                if rng.random() < p:
                    edges.add((u, v))
        return sorted(edges)

    specs = [(20, 0.3, 1), (25, 0.3, 2), (30, 0.25, 3)]
    graphs = [instance_graph(*spec) for spec in specs]

    # 2. Solve each graph using Iterated Local Search on Edge Permutations
    #    Representation: A list of edges. Paths are maximal contiguous sub-sequences
    #    that form valid simple paths in the graph.
    #    Objective: Minimize number of paths.
    
    def solve_graph(edges, time_limit=40.0):
        if not edges:
            return []
        
        # Precompute adjacency for fast checking
        adj = defaultdict(set)
        for u, v in edges:
            adj[u].add(v)
            adj[v].add(u)
        
        # Helper: Evaluate a permutation of edges
        # Returns: (num_paths, paths_list)
        def evaluate(perm):
            if not perm:
                return 0, []
            paths = []
            current_path = [perm[0][0], perm[0][1]]
            
            for i in range(1, len(perm)):
                prev_u, prev_v = perm[i-1]
                curr_u, curr_v = perm[i]
                
                # Check if (prev_u, prev_v) and (curr_u, curr_v) can be joined
                # They share a vertex.
                # Case 1: prev_v == curr_u -> extend current_path
                # Case 2: prev_v == curr_v -> reverse current_path? No, we are building forward.
                # Actually, simpler:
                # The last vertex of current_path is prev_v (assuming we built it that way).
                # Wait, the path is a sequence of vertices.
                # If we have path P = [v0, ..., vk] and next edge e=(u,v).
                # We can append v if vk==u.
                # We can prepend u if vk==v? No, prepending changes the start.
                # Standard path decomposition from a sequence:
                # We maintain a current path.
                # If the next edge shares a vertex with the END of the current path, we can extend.
                # If it shares a vertex with the START, we could extend the other way, but that complicates "current path".
                # Let's stick to: Extend if the next edge is incident to the current head.
                # If not, start a new path.
                
                # Let's refine the evaluation logic for speed.
                # A permutation of edges defines a set of paths by "breaking" the sequence
                # wherever the edges are not "compatible" in a single direction.
                
                # Compatibility:
                # Current path ends at vertex 'head'.
                # Next edge is (a, b).
                # If head == a, we can go to b. New head is b.
                # If head == b, we can go to a. New head is a.
                # Else, break.
                
                # BUT: This greedy evaluation of a permutation might miss paths that could be formed
                # if we reversed a previous segment? No, the permutation fixes the order.
                # However, a path in the graph is undirected.
                # If we have edges e1, e2, e3.
                # e1=(1,2), e2=(2,3), e3=(3,4).
                # Perm: e1, e2, e3.
                # Start 1-2. Head=2.
                # e2=(2,3). Head=2 matches 2. Go to 3. Head=3.
                # e3=(3,4). Head=3 matches 3. Go to 4. Head=4.
                # Path: 1-2-3-4. Count 1.
                
                # What if e2=(3,2)? (stored as sorted tuple (2,3)).
                # Perm: e1=(1,2), e2=(2,3).
                # Head=2. e2 has 2. Match. Go to 3.
                
                # What if the path needs to "turn around"?
                # e1=(1,2), e2=(2,3), e3=(2,4).
                # Perm: e1, e3, e2.
                # Start 1-2. Head=2.
                # e3=(2,4). Head=2 matches 2. Go to 4. Head=4. Path: 1-2-4.
                # e2=(2,3). Head=4. No match. Break.
                # New Path: 2-3.
                # Total 2 paths.
                
                # This evaluation is O(M).
                
                num_paths = 0
                paths_list = []
                current_path_vertices = [perm[0][0], perm[0][1]]
                head = perm[0][1]
                # We need to track the other end too? No, just head is enough for extension.
                # But what if the next edge connects to the TAIL?
                # If we allow extending at the tail, the "current path" concept becomes a deque.
                # Let's try extending at either end to maximize path length.
                
                # Revised Evaluation:
                # Maintain a deque of vertices for the current path.
                # If next edge connects to left_end, appendleft.
                # If next edge connects to right_end, append.
                # If both? (Triangle). Pick one (e.g., right).
                # If neither, start new path.
                
                current_path = [perm[0][0], perm[0][1]]
                left_end = current_path[0]
                right_end = current_path[-1]
                num_paths = 1
                
                for i in range(1, len(perm)):
                    u, v = perm[i]
                    if u == left_end:
                        current_path.insert(0, v)
                        left_end = v
                    elif v == left_end:
                        current_path.insert(0, u)
                        left_end = u
                    elif u == right_end:
                        current_path.append(v)
                        right_end = v
                    elif v == right_end:
                        current_path.append(u)
                        right_end = u
                    else:
                        # Start new path
                        paths_list.append(current_path)
                        num_paths += 1
                        current_path = [u, v]
                        left_end = u
                        right_end = v
                paths_list.append(current_path)
                return num_paths, paths_list

        # Initialize with a random permutation
        best_perm = edges[:]
        random.shuffle(best_perm)
        best_score, best_paths = evaluate(best_perm)
        
        current_perm = best_perm[:]
        current_score, _ = evaluate(current_perm)
        
        t = 1.0
        cooling = 0.995
        
        while time.time() < time.time() + time_limit: # Use absolute deadline check instead
            if time.time() >= deadline:
                break
                
            # Move: Swap two random edges
            i, j = random.sample(range(len(current_perm)), 2)
            current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            
            new_score, _ = evaluate(current_perm)
            
            # Accept if better or with probability exp(-delta/T)
            if new_score < current_score:
                current_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_paths = [p[:] for p in evaluate(current_perm)[1]]
                    best_perm = current_perm[:]
            else:
                # Simulated Annealing acceptance
                delta = new_score - current_score
                if delta == 0:
                    # Accept equal to explore
                    pass 
                else:
                    prob = 1.0 / (1.0 + (new_score - current_score)) # Simple heuristic since scores are integers
                    # Better: exp(-delta/T). Since delta is integer >= 1, and T starts 1.0
                    import math
                    try:
                        p_accept = math.exp(-delta / max(t, 0.01))
                    except OverflowError:
                        p_accept = 0.0
                    
                    if random.random() < p_accept:
                        current_score = new_score
                    else:
                        # Revert
                        current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
            
            t *= cooling
            
            # Restart mechanism: if stuck for too long, reheat or restart
            # Simple restart: if t is very low, reset t and randomize slightly
            if t < 0.05:
                # Random perturbation
                for _ in range(5):
                    i, j = random.sample(range(len(current_perm)), 2)
                    current_perm[i], current_perm[j] = current_perm[j], current_perm[i]
                current_score, _ = evaluate(current_perm)
                t = 1.0

        return best_paths

    all_paths = []
    for i, edges in enumerate(graphs):
        # Allocate time proportional to complexity or just equal slices
        # Total time ~ 145s. 3 graphs. ~45s each.
        # But we have a global deadline.
        time_per_graph = (deadline - time.time()) / (len(graphs) - i)
        if time_per_graph < 1.0:
            # Fallback to greedy if no time
            paths = [[list(e)] for e in edges]
        else:
            paths = solve_graph(edges, time_limit=time_per_graph)
        
        all_paths.append(paths)
        
        # Check deadline
        if time.time() >= deadline - 5.0:
             # Force finish remaining with trivial solution
             for j in range(i+1, len(graphs)):
                 all_paths.append([[list(e)] for e in graphs[j]])
             break

    return all_paths

# EVOLVE-BLOCK-END
