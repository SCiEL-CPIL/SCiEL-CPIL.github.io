"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import random
import time
import math

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Safety margin for 60s limit
    
    n = 15
    # Total possible edges
    total_edges = n * (n - 1) // 2
    edge_indices = list(range(total_edges))
    
    # Map edge index to (u, v)
    # We can generate this on the fly or precompute
    # Precomputing is faster for repeated checks
    edge_map = {}
    idx = 0
    for u in range(n):
        for v in range(u + 1, n):
            edge_map[idx] = (u, v)
            idx += 1
            
    # Initialize with a moderately dense random graph
    # Let's try starting with ~50% density
    initial_density = 0.4
    num_initial = int(total_edges * initial_density)
    random.seed(42) # Reproducibility for the seed, but we can vary if needed
    current_edges = set(random.sample(edge_indices, num_initial))
    
    # Helper to calculate validity of a specific edge addition/removal context
    # A graph is valid if NO simple path s-u-v-t exists such that deg(s) == deg(t).
    # Note: The constraint is on the FULL candidate graph.
    # This is a global constraint. Changing one edge changes degrees of 2 vertices.
    # This changes the validity of many paths.
    
    # Efficient validity check:
    # 1. Compute degrees.
    # 2. For every edge (u, v), check if there exists s in N(u)\{v} and t in N(v)\{u} 
    #    such that s != t and deg(s) == deg(t).
    #    If such a pair exists, the edge (u, v) is "bad" (part of a violating path).
    #    Actually, the constraint is: "Reject if ANY simple three-edge path has endpoints of equal degree".
    #    So if even ONE such path exists, the graph is INVALID.
    
    def is_valid(edge_set):
        # Build adjacency and degrees
        adj = [set() for _ in range(n)]
        deg = [0] * n
        for e in edge_set:
            u, v = edge_map[e]
            adj[u].add(v)
            adj[v].add(u)
            deg[u] += 1
            deg[v] += 1
        
        # Check all edges (u, v) for violating paths s-u-v-t
        # Path is s-u-v-t. s != t. s != v, t != u.
        # We need to check if there exists s in adj[u] - {v} and t in adj[v] - {u} 
        # with s != t and deg[s] == deg[t].
        
        for e in edge_set:
            u, v = edge_map[e]
            # Candidates for s: neighbors of u excluding v
            # Candidates for t: neighbors of v excluding u
            # If deg[s] == deg[t] and s != t, violation.
            
            # Optimization: Group neighbors by degree.
            # For a fixed (u, v):
            # Let S_deg = {deg[s] for s in adj[u] if s != v}
            # Let T_deg = {deg[t] for t in adj[v] if t != u}
            # If there is a common degree d in S_deg and T_deg, 
            # we must ensure that we don't pick the SAME vertex s=t.
            # If s=t, it's not a simple path (s-u-v-s is a cycle of length 3? No, s-u-v-s is a triangle if u-v, v-s, s-u exist. 
            # The path is s-u-v-t. If s=t, it's s-u-v-s. This is a closed walk, not a simple path with distinct endpoints? 
            # "simple three-edge path ... endpoints". Usually implies distinct vertices. 
            # "s != t" is explicitly stated in the prompt: "s != t".
            
            # So, if there is a degree d such that:
            # 1. There exists s in adj[u]\{v} with deg[s]=d
            # 2. There exists t in adj[v]\{u} with deg[t]=d
            # 3. It is NOT the case that the ONLY such s and t are the same vertex.
            
            # Let's collect sets of vertices by degree for neighbors.
            s_by_deg = {}
            for s in adj[u]:
                if s == v: continue
                d = deg[s]
                if d not in s_by_deg:
                    s_by_deg[d] = set()
                s_by_deg[d].add(s)
                
            t_by_deg = {}
            for t in adj[v]:
                if t == u: continue
                d = deg[t]
                if d not in t_by_deg:
                    t_by_deg[d] = set()
                t_by_deg[d].add(t)
                
            # Check intersection of degrees
            for d in s_by_deg:
                if d in t_by_deg:
                    S_set = s_by_deg[d]
                    T_set = t_by_deg[d]
                    # We need s in S_set, t in T_set such that s != t.
                    # This fails ONLY if S_set == T_set == {x} for some x.
                    # i.e., if |S_set| == 1 and |T_set| == 1 and S_set == T_set.
                    if not (len(S_set) == 1 and len(T_set) == 1 and S_set == T_set):
                        return False
                        
        return True

    current_valid = is_valid(current_edges)
    best_edges = current_edges.copy()
    best_score = len(current_edges) if current_valid else -1
    
    if not current_valid:
        # Try to repair by removing edges
        # Simple approach: remove random edges until valid
        while not current_valid and time.time() < deadline:
            e = random.choice(list(current_edges))
            current_edges.remove(e)
            current_valid = is_valid(current_edges)
        best_edges = current_edges.copy()
        best_score = len(current_edges)
    
    # Local Search / Simulated Annealing
    # Moves: Add an edge, Remove an edge
    # We want to maximize edge count.
    # Since validity is a hard constraint, we only accept valid states.
    # This is a greedy hill climbing with restarts if stuck.
    
    # To speed up is_valid, we can cache degree info or use a faster check.
    # However, n=15 is small. is_valid is O(E * (deg_u + deg_v)).
    # Max E ~ 105. deg ~ 14. 105 * 28 ~ 3000 ops. Very fast.
    
    temperature = 1.0
    cooling_rate = 0.999
    
    while time.time() < deadline:
        # Generate a random move
        # Probability of adding vs removing depends on current density
        # If we are valid, we prefer adding.
        
        if current_valid:
            # Try to add an edge
            missing_edges = [i for i in edge_indices if i not in current_edges]
            if missing_edges:
                move_type = 'add'
                edge_to_change = random.choice(missing_edges)
            else:
                move_type = 'remove'
                edge_to_change = random.choice(list(current_edges))
        else:
            # If invalid, we should remove edges to become valid
            move_type = 'remove'
            edge_to_change = random.choice(list(current_edges))
            
        new_edges = current_edges.copy()
        if move_type == 'add':
            new_edges.add(edge_to_change)
        else:
            new_edges.remove(edge_to_change)
            
        new_valid = is_valid(new_edges)
        new_score = len(new_edges) if new_valid else -1
        
        # Acceptance criteria
        if new_valid:
            if new_score > best_score:
                best_edges = new_edges.copy()
                best_score = new_score
            
            # Always accept valid moves that increase score (greedy)
            # Or use SA: accept if better, or with prob exp(-delta/T) if worse?
            # Since we maximize edges, any valid move that adds an edge is better.
            # A valid move that removes an edge is worse in score, but might allow future additions.
            # Let's stick to greedy for valid states: only accept if score increases.
            # If score decreases, we might still accept with small prob to escape local optima?
            # But removing an edge reduces score. Let's just do greedy add-only for valid states?
            # No, sometimes you need to remove an edge to add two others.
            
            # Let's use a simple heuristic:
            # If move increases edge count, accept.
            # If move decreases edge count, accept with probability p (annealing).
            
            if new_score > len(current_edges):
                current_edges = new_edges
                current_valid = True
            else:
                # Worse or equal (equal only if add/remove same count? No, add is +1, remove is -1)
                # So new_score < len(current_edges)
                if random.random() < math.exp(-1.0 / max(temperature, 0.01)):
                    current_edges = new_edges
                    current_valid = True
                    temperature *= cooling_rate
        else:
            # Invalid move. Reject.
            # In SA, we might still accept invalid states to jump over barriers?
            # But the constraint is hard. We can't return invalid.
            # However, exploring invalid states might help find a path to a better valid state?
            # No, because we can't evaluate the "validity" of intermediate invalid states easily for acceptance?
            # Actually, we CAN evaluate validity. We just can't ACCEPT an invalid state as the final answer.
            # But for local search, if we allow invalid intermediate states, we might get stuck.
            # Given the strict "Reject if..." rule, it's safer to stay in the valid space.
            # If we are valid, we reject invalid moves.
            # If we are invalid (shouldn't happen if we start valid and only accept valid), we force removal.
            
            pass
            
        # Cool down
        temperature *= cooling_rate
        
    # Final check and return
    if not is_valid(best_edges):
        # Fallback to a known valid graph (e.g., empty or star)
        # Star is valid:
        # Paths s-0-v-t. deg(0)=14. deg(s)=1, deg(t)=1.
        # s, t are leaves. deg(s)=1, deg(t)=1.
        # Wait. If s and t are distinct leaves, deg(s)=1, deg(t)=1. They are equal!
        # So the star graph IS INVALID?
        # Let's re-read: "Reject if any simple three-edge path has endpoints of equal degree".
        # Star K_{1,14}.
        # Path: leaf1 - center - leaf2 - ... wait, leaf2 is a leaf, it has no other neighbors.
        # A path of length 3 is s-u-v-t.
        # In a star, the only vertex with degree > 1 is the center.
        # So u and v must involve the center?
        # If u is center, v is leaf.
        # Path: s - center - leaf - t.
        # leaf has degree 1. Its only neighbor is center.
        # So t must be a neighbor of leaf. t = center.
        # But t must be distinct from s?
        # Path: s - center - leaf - center.
        # This is not a simple path (center repeats).
        # So in a star graph, there are NO simple paths of length 3.
        # Why?
        # A simple path of length 3 requires 4 distinct vertices s, u, v, t.
        # Edges (s,u), (u,v), (v,t).
        # In a star, any path of length 2 is leaf1-center-leaf2.
        # To extend to length 3, we need leaf2-t.
        # leaf2 is only connected to center. So t=center.
        # Path: leaf1-center-leaf2-center.
        # Vertices: leaf1, center, leaf2, center.
        # Not distinct (center repeats).
        # So no simple path of length 3 exists in a star.
        # Therefore, the star graph is VALID.
        
        # My previous check for star:
        # "deg(s)=1, deg(t)=1. They are equal!"
        # But the path doesn't exist. So it's valid.
        
        # Let's return the star as fallback.
        best_edges = set([edge_map[i] for i in range(14)] if False else [])
        # Generate star edges: (0, 1), (0, 2), ..., (0, 14)
        star_edges = set()
        for v in range(1, n):
            # Find index for (0, v)
            for idx, (u, w) in edge_map.items():
                if (u, w) == (0, v):
                    star_edges.add(idx)
                    break
        best_edges = star_edges
        best_score = len(best_edges)

    # Convert best_edges (set of indices) to list of [u, v]
    result = []
    for e in best_edges:
        u, v = edge_map[e]
        result.append([u, v])
        
    # Ensure types are correct
    # "integers must be Python int"
    # "sequences must be list or tuple"
    # result is a list of lists of ints.
    
    return result
# EVOLVE-BLOCK-END
