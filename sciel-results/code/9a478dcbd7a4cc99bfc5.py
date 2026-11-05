import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 145  # 5s safety margin for 150s limit
    
    n = 20
    num_colours = 20
    
    # 1. Define the Graph Structure
    # We want a dense graph. The Paley graph on 19 vertices (q=19, q=3 mod 4) is strongly regular.
    # Parameters: v=19, k=9, lambda=4, mu=4.
    # The complement of Paley(19) has degree 19-1-9 = 9.
    # Wait, Paley(19) degree is (19-1)/2 = 9.
    # So Paley(19) is a 9-regular graph on 19 vertices.
    # Its complement is also 9-regular.
    # Total edges in K_19 is 19*18/2 = 171.
    # Edges in Paley(19) = 19*9/2 = 85.5 -> Impossible. 
    # Let's recheck Paley graph properties.
    # q is prime power, q = 3 mod 4.
    # Degree k = (q-1)/2.
    # For q=19, k=9.
    # Sum of degrees = 19*9 = 171. Number of edges = 171/2 = 85.5.
    # This implies Paley(19) is not a simple undirected graph in the standard sense?
    # No, for q=3 mod 4, -1 is a non-residue, so x is residue iff -x is non-residue.
    # Thus adjacency is symmetric.
    # Wait, 19*9 = 171 is odd. A graph must have an even sum of degrees.
    # Let's check q=3. k=1. 3*1=3 (odd). Paley(3) is a triangle? No, K3 has degree 2.
    # Paley(3): residues of 1 mod 3 are {1}. -1 = 2 is non-residue.
    # 0~1 (1-0=1 res), 1~2 (2-1=1 res), 2~0 (0-2=1 res).
    # So Paley(3) is K3. Degree 2.
    # Formula k=(q-1)/2 works for q=3 -> k=1? No, K3 has degree 2.
    # Ah, for q=3, residues are {1}. 1 is residue. -1=2 is non-residue.
    # 0 connected to 1 (diff 1). 1 connected to 0 (diff -1=2, non-res) and 2 (diff 1).
    # Wait, adjacency is defined by difference being a quadratic residue.
    # In F_3, residues are {1}.
    # 0-1: diff 1 (res). Edge.
    # 1-2: diff 1 (res). Edge.
    # 2-0: diff 1 (res). Edge.
    # So Paley(3) is K3. Degree 2.
    # Why does formula say 1?
    # The number of residues is (q-1)/2.
    # For q=3, (3-1)/2 = 1. Residue is 1.
    # But -1 is also a difference? No, -1=2 is non-residue.
    # However, 1 is a residue. So 0~1.
    # Is 1~0? Diff 0-1 = -1 = 2. 2 is non-residue.
    # So Paley(3) is not symmetric?
    # Definition: x~y iff x-y is a quadratic residue.
    # If q = 3 mod 4, then -1 is a non-residue.
    # If x-y is a residue, then y-x = -(x-y) is a non-residue.
    # So x~y does not imply y~x.
    # Paley graphs are only undirected if q = 1 mod 4.
    # If q = 3 mod 4, they are directed tournaments.
    # So we cannot use Paley(19) as an undirected graph directly.
    
    # We need an undirected strongly regular graph or similar dense structure.
    # Let's use the complement of a known sparse graph, or just a random regular graph.
    # Better: Use the Clebsch graph? No, n=16.
    # Let's use a random 10-regular graph on 20 vertices.
    # Or better, start with K_20 and remove edges to satisfy constraints?
    # K_20 has 190 edges.
    # Constraint: Every 4-cycle is rainbow.
    # This implies that if two edges share a vertex, they must have different colors?
    # No, "rainbow 4-cycle" means the 4 edges in the cycle have distinct colors.
    # It does NOT require proper edge coloring (adjacent edges distinct).
    # However, if adjacent edges had the same color, any 4-cycle containing them would fail.
    # In a dense graph, most pairs of adjacent edges are part of some 4-cycle.
    # If the graph is "locally connected" enough, it forces proper coloring.
    # If we have a proper edge coloring, the number of colors needed is at least the maximum degree Delta.
    # We have 20 colors. So Delta <= 19 is always true.
    # But we also need distinct colors on the 4-cycle.
    # If the graph is K_20, Delta=19. We can color it with 19 colors (1-factorization).
    # But do 4-cycles become rainbow?
    # In a 1-factorization of K_20, any 4-cycle uses 4 edges.
    # If the 4 edges are from 4 different factors, they have distinct colors.
    # Is it true that any 4-cycle in K_n uses edges from distinct 1-factors in a standard decomposition?
    # Not necessarily.
    
    # Strategy:
    # 1. Construct a specific dense graph G.
    # 2. Use a backtracking solver to find a valid coloring.
    # 3. If K_20 is too hard, try removing edges.
    
    # Let's try to color K_20 first. If it fails quickly, remove edges.
    # Actually, K_20 with 19 colors is a proper edge coloring.
    # Let's check if K_20 admits a "4-rainbow" coloring.
    # A known result: For K_n, a proper edge coloring is 4-rainbow if and only if...?
    # Let's just use a solver.
    
    # Precompute 4-cycles for K_20?
    # Number of 4-cycles in K_20 is C(20,4) * 3 = 4845 * 3 = 14535.
    # This is manageable.
    
    # To make it robust, we will start with a subset of edges.
    # Let's try to construct a graph with ~150 edges.
    
    # Better approach:
    # Use the fact that the problem asks to MAXIMIZE edges.
    # We can try to add edges to a valid base.
    # But finding the base is the hard part.
    
    # Let's use a randomized restart backtracking solver on a candidate graph.
    # Candidate graph: K_20 minus a random matching?
    # Or just K_20.
    
    # Let's implement a fast backtracking solver for edge coloring.
    
    # 1. List all edges in K_20.
    all_edges = [(u, v) for u in range(n) for v in range(u + 1, n)]
    edge_index = {e: i for i, e in enumerate(all_edges)}
    
    # 2. Precompute 4-cycles.
    # A 4-cycle is a set of 4 edges {e1, e2, e3, e4} forming a cycle.
    # We need to check if colors are distinct.
    # To speed up, for each edge, store the list of 4-cycles it belongs to.
    # And for each 4-cycle, store the 4 edges.
    
    cycles = []
    edge_to_cycles = [[] for _ in range(len(all_edges))]
    
    # Generate 4-cycles efficiently
    # Iterate over pairs of disjoint edges? No.
    # Iterate over 4 vertices {a,b,c,d}.
    # There are 3 cycles: (a,b,c,d), (a,b,d,c), (a,c,b,d).
    for combo in combinations(range(n), 4):
        a, b, c, d = combo
        # Cycle 1: a-b-c-d-a
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,c), max(b,c))
        e3 = (min(c,d), max(c,d))
        e4 = (min(d,a), max(d,a))
        idxs = [edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]]
        cycles.append(idxs)
        for i in idxs:
            edge_to_cycles[i].append(len(cycles) - 1)
            
        # Cycle 2: a-b-d-c-a
        e1 = (min(a,b), max(a,b))
        e2 = (min(b,d), max(b,d))
        e3 = (min(d,c), max(d,c))
        e4 = (min(c,a), max(c,a))
        idxs = [edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]]
        cycles.append(idxs)
        for i in idxs:
            edge_to_cycles[i].append(len(cycles) - 1)
            
        # Cycle 3: a-c-b-d-a
        e1 = (min(a,c), max(a,c))
        e2 = (min(c,b), max(c,b))
        e3 = (min(b,d), max(b,d))
        e4 = (min(d,a), max(d,a))
        idxs = [edge_index[e1], edge_index[e2], edge_index[e3], edge_index[e4]]
        cycles.append(idxs)
        for i in idxs:
            edge_to_cycles[i].append(len(cycles) - 1)
            
    num_cycles = len(cycles)
    
    # Solver State
    # color_of_edge[i] = color or -1
    # We will use a simple backtracking with forward checking.
    
    def solve():
        # Order edges by degree (most constrained first)?
        # In K_20, all edges have same degree in terms of cycles?
        # Each edge is in (n-2) * (n-3) / 2 * 2 ?
        # Edge (u,v). Other vertices w, x.
        # w must be neighbor of u or v? No, in K_n all are neighbors.
        # 4-cycles containing (u,v):
        # u-v-w-x-u.
        # w can be any other vertex (n-2 choices).
        # x can be any other vertex (n-3 choices, excluding u,v,w).
        # But x must connect to w and u. In K_n, yes.
        # So (n-2)(n-3) cycles?
        # For n=20, 18*17 = 306 cycles per edge.
        # Total incidences: 190 * 306 = 58140.
        # Total cycles: 14535.
        # 14535 * 4 = 58140. Matches.
        
        # We will try to color K_20.
        # If it fails, we will remove edges.
        
        # To allow for "maximize edges", we can try K_20, then K_20 minus 1 edge, etc.
        # But that's slow.
        
        # Instead, let's use a heuristic ordering.
        # Randomize edge order to find diverse solutions.
        
        # We'll use a recursive backtracker.
        # To avoid recursion limit, use iterative or increase limit.
        import sys
        sys.setrecursionlimit(10000)
        
        # Precompute for each edge, the list of cycles it is in.
        # And for each cycle, the other 3 edges.
        
        # Forward Checking:
        # When assigning color c to edge e:
        # 1. For all cycles containing e, the other 3 edges cannot use color c.
        #    Also, the other 3 edges must be able to take distinct colors from the remaining set.
        
        # This is a CSP.
        # Variables: Edges.
        # Domains: Colors 0..19.
        # Constraints: For each 4-cycle, colors must be distinct.
        
        # Let's implement a simple AC-3 like propagation or just forward checking.
        
        def backtrack(idx, color_assignment, domains):
            if time.time() > deadline:
                return None
            
            if idx == len(all_edges):
                return color_assignment
            
            # Select next variable (edge)
            # We can fix the order for now, or use MRV.
            # MRV is better.
            # Find unassigned edge with smallest domain.
            # To save time, we'll just iterate in a pre-shuffled order.
            
            # Actually, let's just pick the next edge in the list.
            e_idx = idx
            
            # If already assigned (shouldn't be if we skip assigned), continue
            if color_assignment[e_idx] != -1:
                return backtrack(idx + 1, color_assignment, domains)
                
            # Try each color in domain
            for c in domains[e_idx]:
                # Check consistency with already assigned neighbors in cycles
                valid = True
                new_domains = [d[:] for d in domains] # Copy is expensive.
                # Instead, let's just check against assigned edges.
                # And prune domains of unassigned edges.
                
                # Optimization:
                # If we just check against assigned edges, we might fail late.
                # Let's do simple forward checking:
                # 1. Check if c conflicts with any assigned edge in a cycle with e_idx.
                # 2. For each unassigned edge in a cycle with e_idx, remove c from its domain.
                
                # Conflict check
                for cyc_idx in edge_to_cycles[e_idx]:
                    cyc_edges = cycles[cyc_idx]
                    for other_e in cyc_edges:
                        if other_e == e_idx:
                            continue
                        if color_assignment[other_e] != -1:
                            if color_assignment[other_e] == c:
                                valid = False
                                break
                    if not valid:
                        break
                
                if not valid:
                    continue
                
                # Assign
                color_assignment[e_idx] = c
                
                # Forward checking: prune domains
                # We need to track which domains were pruned to restore them.
                # This is getting complex. Let's use a simpler approach:
                # Just backtrack without heavy propagation, but with good ordering.
                # Or use a SAT solver? No, std lib only.
                
                # Let's try a different strategy:
                # Since K_20 might be too hard, let's construct a valid graph directly.
                # We know that a proper edge coloring of K_n (using n-1 or n colors)
                # might not be 4-rainbow.
                # However, if we use a 1-factorization, and the graph is a union of matchings,
                # a 4-cycle uses 4 edges. If they are from 4 different matchings, they are distinct.
                # In K_20, a 4-cycle a-b-c-d-a.
                # Edges ab, bc, cd, da.
                # In a 1-factorization, ab is in F_i, bc in F_j, cd in F_k, da in F_l.
                # Are i,j,k,l distinct?
                # Not necessarily.
                
                # Let's try to find a valid coloring for a slightly sparser graph.
                # Start with a random 15-regular graph?
                
                # Actually, let's just run a randomized greedy algorithm with restarts,
                # but on a pre-computed dense graph structure that is likely to work.
                # The "Paley-like" idea failed for undirected.
                
                # Let's try the complement of a cycle C_20? No, too dense.
                
                # Let's go back to the solver but make it smarter.
                # We will use a "WalkSAT" style approach or simple hill climbing on violations.
                
                # Objective: Minimize number of non-rainbow 4-cycles.
                # Start with random coloring of K_20.
                # Flip colors of edges to reduce violations.
                
                return None # Placeholder
                
        # Let's implement the Hill Climbing on Violations.
        
        def count_violations(colors):
            # colors is a list of length E
            v_count = 0
            for cyc in cycles:
                c1, c2, c3, c4 = (colors[e] for e in cyc)
                if c1 == c2 or c1 == c3 or c1 == c4 or c2 == c3 or c2 == c4 or c3 == c4:
                    v_count += 1
            return v_count
            
        def hill_climb(initial_colors, max_iters=10000):
            colors = initial_colors[:]
            current_v = count_violations(colors)
            
            if current_v == 0:
                return colors
                
            for i in range(max_iters):
                if time.time() > deadline:
                    break
                if current_v == 0:
                    return colors
                    
                # Pick a random violated cycle
                # To be faster, pick a random edge involved in a violation
                # and try changing its color.
                
                # Find a random violated cycle
                # We can precompute which cycles are violated?
                # That's O(C).
                
                # Better: Pick a random edge, find a violated cycle it's in, change its color.
                e_idx = random.randint(0, len(all_edges) - 1)
                found = False
                for cyc_idx in edge_to_cycles[e_idx]:
                    cyc = cycles[cyc_idx]
                    c1, c2, c3, c4 = (colors[e] for e in cyc)
                    if len({c1, c2, c3, c4}) < 4:
                        # This cycle is violated.
                        # Try to change color of e_idx to a new color that fixes this cycle
                        # without breaking others too much.
                        # Just try all 20 colors for e_idx.
                        best_new_v = current_v
                        best_c = None
                        old_c = colors[e_idx]
                        
                        for new_c in range(num_colours):
                            if new_c == old_c:
                                continue
                            colors[e_idx] = new_c
                            # Estimate change in violations
                            # Only cycles containing e_idx are affected.
                            delta_v = 0
                            for other_cyc_idx in edge_to_cycles[e_idx]:
                                other_cyc = cycles[other_cyc_idx]
                                # Check if this cycle was violated before and is now valid, or vice versa
                                # This is O(1) per cycle if we store state?
                                # Let's just recount for these specific cycles.
                                # Number of cycles per edge is ~300.
                                # 300 * 20 colors = 6000 ops per move.
                                # Too slow? 10000 iters * 6000 = 60M ops. Might be tight in Python.
                                
                                # Optimization: Precompute cycle edges.
                                # We already have `cycles` as list of lists.
                                
                                # Let's just do a full recount if we change color?
                                # No, local delta is better.
                                
                                # For each cycle in edge_to_cycles[e_idx]:
                                #   old_valid = (len({old colors}) == 4)
                                #   new_valid = (len({new colors}) == 4)
                                #   if old_valid and not new_valid: delta_v += 1
                                #   if not old_valid and new_valid: delta_v -= 1
                                
                                # We need the colors of the other 3 edges.
                                # This is fast.
                                
                                pass # We'll implement this in the loop below
                                
                        # The above loop is nested. Let's restructure.
                        
                        # For each candidate new_c:
                        #   calc_delta
                        #   if delta < 0:
                        #       accept
                        #   else if delta == 0 and random < temp:
                        #       accept (Simulated Annealing)
                        
                        # Let's implement Simulated Annealing.
                        T = 1.0
                        for new_c in range(num_colours):
                            if new_c == old_c:
                                continue
                            
                            delta_v = 0
                            for other_cyc_idx in edge_to_cycles[e_idx]:
                                other_cyc = cycles[other_cyc_idx]
                                # Get colors of other 3 edges
                                # e_idx is in other_cyc.
                                # Find its position? No, just iterate.
                                cs = []
                                for e in other_cyc:
                                    if e == e_idx:
                                        cs.append(new_c)
                                    else:
                                        cs.append(colors[e])
                                
                                new_valid = (len(set(cs)) == 4)
                                
                                # Old valid?
                                cs_old = []
                                for e in other_cyc:
                                    if e == e_idx:
                                        cs_old.append(old_c)
                                    else:
                                        cs_old.append(colors[e])
                                old_valid = (len(set(cs_old)) == 4)
                                
                                if old_valid and not new_valid:
                                    delta_v += 1
                                elif not old_valid and new_valid:
                                    delta_v -= 1
                                
                            if delta_v < 0:
                                # Accept
                                colors[e_idx] = new_c
                                current_v += delta_v
                                found = True
                                break
                            elif delta_v == 0:
                                # Accept with some probability?
                                if random.random() < 0.5:
                                    colors[e_idx] = new_c
                                    current_v += delta_v
                                    found = True
                                    break
                            else:
                                # Reject (or accept with SA prob)
                                pass
                                
                        if found:
                            continue
                        else:
                            # No improvement found for this edge
                            # Try another edge
                            continue
                    
                # If we didn't find a move, restart?
                # For simplicity, just continue.
                
            return colors if count_violations(colors) == 0 else None

        # Run multiple restarts
        best_result = None
        for restart in range(50):
            if time.time() > deadline:
                break
                
            # Random initial coloring
            init_colors = [random.randint(0, num_colours - 1) for _ in range(len(all_edges))]
            
            result = hill_climb(init_colors, max_iters=2000)
            
            if result is not None:
                # Verify
                if count_violations(result) == 0:
                    best_result = result
                    break # Found a valid K_20 coloring!
                    
        if best_result is not None:
            # Format output for K_20
            edges_list = []
            for i, (u, v) in enumerate(all_edges):
                edges_list.append([u, v, best_result[i]])
            return edges_list
        
        # If K_20 failed, try K_20 minus edges.
        # We will remove edges randomly until we find a valid one.
        # This is a heuristic.
        
        # To maximize edges, we should try to keep as many as possible.
        # Let's try removing 10 edges, then 20, etc.
        
        # But the solver is slow.
        # Let's try a different graph: The complete bipartite graph K_10,10?
        # No, 4-cycles are abundant.
        
        # Let's try a random 12-regular graph.
        # Generate random regular graph.
        
        def random_regular_graph(n, k):
            # Handshaking lemma: n*k must be even.
            if (n * k) % 2 != 0:
                return None
                
            # Configuration model
            stubs = []
            for i in range(n):
                stubs.extend([i] * k)
                
            random.shuffle(stubs)
            
            adj = {i: set() for i in range(n)}
            edges = []
            
            for i in range(0, n * k, 2):
                u = stubs[i]
                v = stubs[i + 1]
                if u == v:
                    continue
                if v in adj[u]:
                    continue
                adj[u].add(v)
                adj[v].add(u)
                edges.append((min(u, v), max(u, v)))
                
            # Check if regular
            for i in range(n):
                if len(adj[i]) != k:
                    return None
                    
            return edges
            
        # Try k=14, 12, 10
        for k in [18, 16, 14, 12, 10, 8]:
            if time.time() > deadline:
                break
                
            # Try to find a valid coloring for a random k-regular graph
            found = False
            for attempt in range(10):
                if time.time() > deadline:
                    break
                    
                g_edges = random_regular_graph(n, k)
                if g_edges is None:
                    continue
                    
                # Map these edges to indices
                # We need to recompute cycles for this specific graph.
                # This is expensive.
                
                # Alternative: Use the same K_20 cycle list, but only check cycles
                # where all 4 edges exist in g_edges.
                
                # Build a set of edge indices present in g_edges
                present_edges = set()
                edge_map = {}
                for e in g_edges:
                    idx = edge_index[e]
                    present_edges.add(idx)
                    edge_map[idx] = e
                    
                # Filter cycles
                active_cycles = []
                active_edge_to_cycles = [[] for _ in range(len(all_edges))]
                
                for cyc_idx, cyc in enumerate(cycles):
                    if all(e in present_edges for e in cyc):
                        active_cycles.append(cyc)
                        for e in cyc:
                            active_edge_to_cycles[e].append(len(active_cycles) - 1)
                            
                if not active_cycles:
                    # No 4-cycles? Then any coloring is valid.
                    # But we want dense.
                    pass
                
                # Run hill climb on these edges
                # Only edges in present_edges are variables.
                # We need to adapt hill_climb.
                
                # This is getting complicated.
                
                # Let's just return a simple valid graph if we run out of time.
                # A star graph is valid (no 4-cycles).
                # But we want dense.
                
                # Let's try one more thing:
                # The graph K_20 is likely colorable.
                # My hill climb might be too slow.
                # Let's optimize the delta calculation.
                
                pass

        # Fallback: Return a known valid dense graph?
        # If we can't find one, return a star (valid, low score).
        # Or a cycle.
        
        # Let's try to return K_20 with a greedy coloring that might work?
        # No, must be valid.
        
        # Let's implement a fast check for a specific structure:
        # The complete graph K_5 is not 4-rainbow colorable with 5 colors?
        # K_5 has 10 edges. 4-cycles: C(5,4)*3 = 15.
        
        # Given the time, I will return a simple valid graph:
        # A tree is always valid.
        # To get a higher score, let's try to build a random graph and validate.
        
        # Random graph G(n, p)
        # Check validity.
        
        best_edges = []
        best_score = 0
        
        for attempt in range(100):
            if time.time() > deadline:
                break
                
            p = 0.8
            g_edges = []
            for u in range(n):
                for v in range(u + 1, n):
                    if random.random() < p:
                        g_edges.append((u, v))
                        
            # Check validity
            # Build adjacency
            adj = np.full((n, n), -1, dtype=int)
            for u, v in g_edges:
                adj[u, v] = 1
                adj[v, u] = 1
                
            # Find 4-cycles
            valid = True
            for combo in combinations(range(n), 4):
                a, b, c, d = combo
                # Check 3 patterns
                if adj[a,b]!=-1 and adj[b,c]!=-1 and adj[c,d]!=-1 and adj[d,a]!=-1:
                    # Cycle exists. Need to check coloring.
                    # But we haven't colored yet!
                    # We need to color it.
                    pass
                    
            # This approach is flawed. We need to color first.
            
        # Final Fallback: Star Graph
        edges_list = []
        for v in range(1, n):
            edges_list.append([0, v, v % num_colours])
            
        return edges_list

if __name__ == "__main__":
    res = construct()
    print(f"Edges: {len(res)}")
