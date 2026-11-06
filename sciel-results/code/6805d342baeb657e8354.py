<<<<<<< SEARCH
import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 115s.
    deadline = start_time + 115.0
    
    n = 20
    best_count = 0
    best_adj = [0] * n
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u, v) maintains K2,3-freeness.
        adj is a list of integers (bitmasks) where adj[i] has bit j set if (i,j) is an edge.
        
        Constraints for adding (u, v):
        1. u and v must not already be connected.
        2. The number of common neighbors of u and v must be <= 2.
        3. For any neighbor w of u (w != v), |N(v) \cap N(w)| <= 1.
        4. For any neighbor w of v (w != u), |N(u) \cap N(w)| <= 1.
        """
        # 1. Check if edge exists
        if (adj[u] >> v) & 1:
            return False
            
        # 2. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # Fast pre-check: if u has any neighbor w such that v and w already share 2+ common neighbors,
        # adding (u,v) would make u the 3rd common neighbor. Similarly for v.
        # We can check this more efficiently: for each w in N(u)\{v}, check popcount(adj[v] & adj[w]) > 1.
        # Pre-check: if |N(u) \ N(v)| is large and adj[v] is dense, likely to fail.
        # Actually, let's just do the check but with an early-exit optimization.
        
        # 3. Check constraints on neighbors of u
        w_mask = adj[u] & ~(1 << v)  # exclude v from consideration
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 4. Check constraints on neighbors of v
        w_mask = adj[v] & ~(1 << u)  # exclude u from consideration
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    missing.append((i, j))
        return missing

    def get_present_edges(adj):
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        return present

    def greedy_fill(adj, count):
        """
        Try to add as many edges as possible from the missing list.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
                    # Don't break, keep trying to add more in this pass
        return count

    def simulated_annealing(adj, count):
        """
        Simulated annealing to escape local optima.
        Uses time-based cooling instead of step-based for consistent behavior.
        """
        if time.time() > deadline - 5:
            return count
        
        sa_start = time.time()
        sa_duration = 8.0  # seconds per SA phase
        sa_end = min(sa_start + sa_duration, deadline - 2)
        
        T = 1.5
        min_T = 0.01
        # Use time-based cooling: T decreases exponentially over sa_duration
        total_time = sa_duration
        cooling_rate = (min_T / T) ** (1.0 / total_time) if total_time > 0 else 1.0
        
        steps_since_improve = 0
        max_steps_no_improve = 200
        
        while time.time() < sa_end and steps_since_improve < max_steps_no_improve:
            # Pick a random move
            move_type = random.random()
            
            if move_type < 0.5:
                # Try to add a missing edge
                missing = get_missing_edges(adj)
                if missing:
                    u, v = random.choice(missing)
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
            elif move_type < 0.8:
                # Remove a present edge and then refill (kick move)
                present = get_present_edges(adj)
                if present:
                    u, v = random.choice(present)
                    old_count = count
                    remove_edge(adj, u, v)
                    count -= 1
                    
                    # Try to greedily add back edges
                    count = greedy_fill(adj, count)
                    if count > old_count:
                        steps_since_improve = 0
                    elif count == old_count:
                        # Neutral, accept with probability based on T
                        if random.random() < T:
                            steps_since_improve = 0
                        else:
                            # Revert: we can't easily revert greedy_fill, so accept
                            steps_since_improve += 1
                    else:
                        # Got worse, but accept with probability T to escape
                        if random.random() < T:
                            steps_since_improve = 0
                        else:
                            steps_since_improve += 1
            else:
                # Swap: Remove one, add one specific missing
                present = get_present_edges(adj)
                missing = get_missing_edges(adj)
                if present and missing:
                    u1, v1 = random.choice(present)
                    u2, v2 = random.choice(missing)
                    
                    # Temporarily remove
                    remove_edge(adj, u1, v1)
                    count -= 1
                    
                    # Try to add
                    if is_valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        count += 1
                        # Count stays same, but structure changed.
                        # Accept neutral moves to explore.
                        steps_since_improve = 0
                    else:
                        # Revert
                        add_edge(adj, u1, v1)
                        count += 1
                        steps_since_improve += 1
            
            # Cool based on elapsed time
            elapsed = time.time() - sa_start
            T = max(min_T, T * cooling_rate ** (1.0 / 60.0))  # approximate per-step cooling
            
        # Final greedy fill
        count = greedy_fill(adj, count)
        
        return count

    def smart_initialization():
        """
        Initialize with a dense starting graph: start with all edges, then remove
        edges that violate K2,3-freeness. This gives a much denser starting point
        than random greedy fill.
        """
        adj = [0] * n
        count = 0
        # Start with a random subset to avoid symmetric traps
        # Use a random permutation and greedily add
        random.shuffle(all_edges)
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        return adj, count

    def smart_initialization_v2():
        """
        Alternative: start with complement of a random matching, then greedily add.
        This starts with ~190 edges minus a few, giving a very dense seed.
        """
        adj = [0] * n
        count = 0
        # Start with all edges
        for i in range(n):
            for j in range(i + 1, n):
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
        count = n * (n - 1) // 2
        
        # Now remove edges that cause violations.
        # A violation is a pair (u,v) with 3+ common neighbors.
        # We need to remove enough edges to fix all violations.
        # Strategy: for each violating pair, remove one edge incident to a common neighbor.
        
        changed = True
        max_iter = 50
        it = 0
        while changed and it < max_iter and time.time() < deadline:
            changed = False
            it += 1
            # Find a violating pair
            for u in range(n):
                for v in range(u + 1, n):
                    common = adj[u] & adj[v]
                    if common.bit_count() > 2:
                        # Remove a random edge to break the violation
                        # Remove edge (u, v) or an edge from a common neighbor
                        # Let's remove a random edge among the common neighbors' connections
                        c_list = []
                        cm = common
                        while cm:
                            lsb = cm & -cm
                            w = lsb.bit_length() - 1
                            c_list.append(w)
                            cm ^= lsb
                        # Remove a random edge incident to one of the common neighbors
                        w = random.choice(c_list)
                        # Remove edge (w, u) or (w, v)
                        if random.random() < 0.5:
                            remove_edge(adj, w, u)
                        else:
                            remove_edge(adj, w, v)
                        count -= 1
                        changed = True
                        break
                if changed:
                    break
        return adj, count

    # Run multiple random restarts
    restarts = 0
    while time.time() < deadline - 10:
        # Alternate between initialization strategies
        if restarts % 3 == 0:
            # Random greedy fill
            random.shuffle(all_edges)
            adj = [0] * n
            count = 0
            for u, v in all_edges:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
        elif restarts % 3 == 1:
            # Dense start with violation removal
            adj, count = smart_initialization_v2()
        else:
            # Random greedy with different seed
            random.shuffle(all_edges)
            adj = [0] * n
            count = 0
            for u, v in all_edges:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
        
        # Local search to fill in missing edges
        count = greedy_fill(adj, count)
        
        # Simulated annealing
        count = simulated_annealing(adj, count)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
            
        restarts += 1
        
        # If we are running out of time, break early
        if time.time() > deadline - 15:
            break

    # Construct the output list of edges
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges
=======
import time
import random
import sys

def construct():
    start_time = time.time()
    # Safety margin: Evaluator kills at 150s. We stop at 115s.
    deadline = start_time + 115.0
    
    n = 20
    best_count = 0
    best_adj = [0] * n
    
    # Pre-compute all possible edges
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = len(all_edges)  # 190
    
    # Edge index mapping
    edge_idx = {}
    for idx, (u, v) in enumerate(all_edges):
        edge_idx[(u, v)] = idx
        edge_idx[(v, u)] = idx
    
    def is_valid_add(adj, u, v):
        """
        Check if adding edge (u, v) maintains K2,3-freeness.
        adj is a list of integers (bitmasks) where adj[i] has bit j set if (i,j) is an edge.
        
        Constraints for adding (u, v):
        1. u and v must not already be connected.
        2. The number of common neighbors of u and v must be <= 2.
        3. For any neighbor w of u (w != v), |N(v) \cap N(w)| <= 1.
        4. For any neighbor w of v (w != u), |N(u) \cap N(w)| <= 1.
        """
        # 1. Check if edge exists
        if (adj[u] >> v) & 1:
            return False
            
        # 2. Check common neighbors of u and v
        common_uv = adj[u] & adj[v]
        if common_uv.bit_count() > 2:
            return False
        
        # 3. Check constraints on neighbors of u
        w_mask = adj[u] & ~(1 << v)  # exclude v from consideration
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_vw = adj[v] & adj[w]
            if common_vw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        # 4. Check constraints on neighbors of v
        w_mask = adj[v] & ~(1 << u)  # exclude u from consideration
        while w_mask:
            lsb = w_mask & -w_mask
            w = lsb.bit_length() - 1
            common_uw = adj[u] & adj[w]
            if common_uw.bit_count() > 1:
                return False
            w_mask ^= lsb
            
        return True

    def add_edge(adj, u, v):
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)

    def remove_edge(adj, u, v):
        adj[u] &= ~(1 << v)
        adj[v] &= ~(1 << u)

    def get_missing_edges(adj):
        missing = []
        for i in range(n):
            for j in range(i + 1, n):
                if not (adj[i] >> j) & 1:
                    missing.append((i, j))
        return missing

    def get_present_edges(adj):
        present = []
        for i in range(n):
            mask = adj[i]
            while mask:
                lsb = mask & -mask
                j = lsb.bit_length() - 1
                if i < j:
                    present.append((i, j))
                mask ^= lsb
        return present

    def greedy_fill(adj, count):
        """
        Try to add as many edges as possible from the missing list.
        """
        improved = True
        while improved and time.time() < deadline:
            improved = False
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            for u, v in missing:
                if is_valid_add(adj, u, v):
                    add_edge(adj, u, v)
                    count += 1
                    improved = True
        return count

    def simulated_annealing(adj, count):
        """
        Simulated annealing to escape local optima.
        """
        if time.time() > deadline - 5:
            return count
        
        sa_start = time.time()
        sa_duration = 8.0
        sa_end = min(sa_start + sa_duration, deadline - 2)
        
        T = 1.5
        min_T = 0.01
        total_time = sa_duration
        cooling_rate = (min_T / T) ** (1.0 / total_time) if total_time > 0 else 1.0
        
        steps_since_improve = 0
        max_steps_no_improve = 200
        
        while time.time() < sa_end and steps_since_improve < max_steps_no_improve:
            move_type = random.random()
            
            if move_type < 0.5:
                missing = get_missing_edges(adj)
                if missing:
                    u, v = random.choice(missing)
                    if is_valid_add(adj, u, v):
                        add_edge(adj, u, v)
                        count += 1
                        steps_since_improve = 0
                    else:
                        steps_since_improve += 1
            elif move_type < 0.8:
                present = get_present_edges(adj)
                if present:
                    u, v = random.choice(present)
                    old_count = count
                    remove_edge(adj, u, v)
                    count -= 1
                    count = greedy_fill(adj, count)
                    if count > old_count:
                        steps_since_improve = 0
                    elif count == old_count:
                        if random.random() < T:
                            steps_since_improve = 0
                        else:
                            steps_since_improve += 1
                    else:
                        if random.random() < T:
                            steps_since_improve = 0
                        else:
                            steps_since_improve += 1
            else:
                present = get_present_edges(adj)
                missing = get_missing_edges(adj)
                if present and missing:
                    u1, v1 = random.choice(present)
                    u2, v2 = random.choice(missing)
                    remove_edge(adj, u1, v1)
                    count -= 1
                    if is_valid_add(adj, u2, v2):
                        add_edge(adj, u2, v2)
                        count += 1
                        steps_since_improve = 0
                    else:
                        add_edge(adj, u1, v1)
                        count += 1
                        steps_since_improve += 1
            
            elapsed = time.time() - sa_start
            T = max(min_T, T * cooling_rate ** (1.0 / 60.0))
        
        count = greedy_fill(adj, count)
        return count

    # ===================== CONFLICT GRAPH / MAX INDEPENDENT SET =====================
    
    # Build the conflict graph on the m=190 potential edges.
    # Two edges e1=(a,b) and e2=(c,d) conflict if adding both to an empty graph
    # would create a K2,3, OR if they are the same edge (trivially incompatible).
    # Actually, for MIS formulation: we want a subset of edges with no K2,3.
    # Two edges conflict if they cannot both be in a K2,3-free graph.
    # 
    # Two edges (u,v) and (u',v') conflict if:
    #   - They share a vertex AND the resulting configuration would force a K2,3
    #   - More precisely: edges e1=(a,b), e2=(c,d) conflict if there exist
    #     vertices x,y such that x,y are both adjacent to both a and b (K2,3 with
    #     {a,b} on one side and {x,y,?} on other). 
    #
    # Actually the correct conflict: edges e1 and e2 conflict if they are part of
    # a K2,3 together with some third edge. But for a pairwise conflict graph,
    # the standard approach is:
    #   e1=(a,b) and e2=(c,d) conflict if |{a,b} ∩ {c,d}| >= 1 AND the pair
    #   creates a situation where 3 common neighbors are forced.
    #
    # Simpler correct approach: e1=(a,b) and e2=(c,d) conflict if there exists
    # a vertex w such that w is adjacent to both endpoints of e1 AND both endpoints
    # of e2, i.e., w is a common neighbor of {a,b} and {c,d}. But that requires
    # the edges to be present.
    #
    # For the MIS formulation to work correctly, we need: a set of edges is
    # K2,3-free if and only if no two edges in the set conflict (where conflict
    # means they form part of a K2,3 with some other edges). This is NOT a simple
    # pairwise property in general.
    #
    # However, there IS a clean pairwise conflict: edges (u,v) and (u',v') conflict
    # if they share exactly one vertex, say u=u', and v != v', and the number of
    # common neighbors of v and v' (excluding u) is >= 2. Then {v,v'} would have
    # u plus those 2 common neighbors as common neighbors = 3, forming K2,3.
    #
    # Actually let me think more carefully. K2,3 has two vertices on one side
    # (say a,b) and three on the other (c,d,e), with all 6 edges present.
    # So a K2,3 exists if there are vertices a,b with at least 3 common neighbors.
    #
    # For the conflict graph: edge e1=(a,b) and edge e2=(c,d) are in conflict if
    # they cannot coexist in any K2,3-free graph. This happens when:
    #   - a=c and b=d (same edge)
    #   - The edges are such that their presence forces some pair to have 3 common neighbors
    #
    # The cleanest pairwise conflict: e1=(a,b) and e2=(a,c) (sharing vertex a).
    # They conflict if b and c have >= 2 common neighbors other than a. Because
    # then {b,c} would have a and those 2+ common neighbors as common neighbors.
    #
    # Also: e1=(a,b) and e2=(c,d) with all four distinct. They conflict if there
    # exists a vertex w adjacent to both a and b, and also adjacent to both c and d.
    # But that's not pairwise-determinable without knowing other edges.
    #
    # Given the complexity, let me use a different approach:
    # Use the conflict graph where e1 and e2 conflict if they share a vertex and
    # the non-shared endpoints have >= 2 common neighbors. This captures the
    # most important conflicts. Then use branch-and-bound on this conflict graph
    # as a heuristic, and verify the result is truly K2,3-free.

    # Build adjacency for vertices
    # For each pair of vertices (a,b), compute common neighbors
    common_neighbors = [[0]*n for _ in range(n)]
    for a in range(n):
        for b in range(a+1, n):
            # common neighbors of a and b
            cn = 0
            for w in range(n):
                if w != a and w != b:
                    cn |= (1 << w)
            common_neighbors[a][b] = cn
            common_neighbors[b][a] = cn
    
    # Build conflict bitmasks for the m edges
    # conflict[i] is a bitmask of edges that conflict with edge i
    conflict = [0] * m
    for i in range(m):
        u1, v1 = all_edges[i]
        for j in range(i+1, m):
            u2, v2 = all_edges[j]
            # Check if edges conflict
            # Case 1: share one vertex
            if u1 == u2 and v1 != v2:
                # shared vertex is u1, non-shared are v1, v2
                # common neighbors of v1 and v2 excluding u1
                cn = common_neighbors[v1][v2] & ~(1 << u1)
                if cn.bit_count() >= 2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
            elif u1 == v2 and v1 != u2:
                # shared vertex is u1=v2, non-shared are v1, u2
                cn = common_neighbors[v1][u2] & ~(1 << u1)
                if cn.bit_count() >= 2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
            elif v1 == u2 and v1 != v2:
                # shared vertex is v1=u2, non-shared are u1, v2
                cn = common_neighbors[u1][v2] & ~(1 << v1)
                if cn.bit_count() >= 2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
            elif v1 == v2 and u1 != u2:
                # shared vertex is v1=v2, non-shared are u1, u2
                cn = common_neighbors[u1][u2] & ~(1 << v1)
                if cn.bit_count() >= 2:
                    conflict[i] |= (1 << j)
                    conflict[j] |= (1 << i)
            # Case 2: all four vertices distinct - no direct pairwise conflict
            # (conflicts only arise through third edges)
    
    # Branch and bound for maximum independent set on the conflict graph
    # An independent set in the conflict graph = a set of edges with no pairwise conflicts
    # This is a sufficient (not necessary) condition for K2,3-freeness, but we'll
    # verify at the end.
    
    def branch_and_bound_mis():
        """Find a large independent set in the conflict graph using branch-and-bound."""
        global best_count, best_adj
        if time.time() > deadline - 30:
            return
        
        # Use Python int as bitset for m=190 edges
        full_mask = (1 << m) - 1
        
        # Greedy initial solution
        greedy_set = 0
        remaining = full_mask
        while remaining:
            # Pick edge with minimum degree in remaining subgraph
            best_e = -1
            best_deg = m + 1
            temp = remaining
            while temp:
                lsb = temp & -temp
                e = lsb.bit_length() - 1
                deg = (conflict[e] & remaining).bit_count()
                if deg < best_deg:
                    best_deg = deg
                    best_e = e
                temp ^= lsb
            greedy_set |= (1 << best_e)
            remaining &= ~(1 << best_e)
            remaining &= ~conflict[best_e]
        
        greedy_count = greedy_set.bit_count()
        
        # Branch and bound
        best_bnb_count = greedy_count
        best_bnb_set = greedy_set
        
        # Order edges by degree for better branching
        edge_order = sorted(range(m), key=lambda e: conflict[e].bit_count())
        
        def bnb(available, current_set, current_count, depth):
            nonlocal best_bnb_count, best_bnb_set
            
            if time.time() > deadline - 25:
                return
            
            # Upper bound: current_count + popcount(available)
            ub = current_count + available.bit_count()
            if ub <= best_bnb_count:
                return
            
            if available == 0:
                if current_count > best_bnb_count:
                    best_bnb_count = current_count
                    best_bnb_set = current_set
                return
            
            # Pick a vertex to branch on (highest degree in available)
            best_v = -1
            best_v_deg = -1
            temp = available
            while temp:
                lsb = temp & -temp
                v = lsb.bit_length() - 1
                deg = (conflict[v] & available).bit_count()
                if deg > best_v_deg:
                    best_v_deg = deg
                    best_v = v
                temp ^= lsb
            
            # Branch 1: include v
            new_available = available & ~(1 << best_v) & ~conflict[best_v]
            bnb(new_available, current_set | (1 << best_v), current_count + 1, depth + 1)
            
            # Branch 2: exclude v
            new_available = available & ~(1 << best_v)
            bnb(new_available, current_set, current_count, depth + 1)
        
        bnb(full_mask, 0, 0, 0)
        
        # Convert best_bnb_set to adjacency
        adj = [0] * n
        count = 0
        temp = best_bnb_set
        while temp:
            lsb = temp & -temp
            e = lsb.bit_length() - 1
            u, v = all_edges[e]
            add_edge(adj, u, v)
            count += 1
            temp ^= lsb
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Run the MIS-based approach
    branch_and_bound_mis()
    
    # Now polish with local search
    if best_count > 0 and time.time() < deadline - 15:
        # Try to improve the MIS solution with greedy fill and SA
        adj = best_adj[:]
        count = best_count
        count = greedy_fill(adj, count)
        count = simulated_annealing(adj, count)
        if count > best_count:
            best_count = count
            best_adj = adj[:]
    
    # Run a few more random restarts with local search if time permits
    restarts = 0
    while time.time() < deadline - 10:
        random.shuffle(all_edges)
        adj = [0] * n
        count = 0
        for u, v in all_edges:
            if is_valid_add(adj, u, v):
                add_edge(adj, u, v)
                count += 1
        
        count = greedy_fill(adj, count)
        count = simulated_annealing(adj, count)
        
        if count > best_count:
            best_count = count
            best_adj = adj[:]
        
        restarts += 1
        if time.time() > deadline - 15:
            break

    # Construct the output list of edges
    edges = []
    for i in range(n):
        mask = best_adj[i]
        while mask:
            lsb = mask & -mask
            j = lsb.bit_length() - 1
            if i < j:
                edges.append([i, j])
            mask ^= lsb
            
    return edges
>>>>>>> REPLACE

Now let me also update the `__main__` block to remain consistent:

