import time
import random
import numpy as np

def construct():
    n = 40
    target_edges = 401
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin

    def calc_books(A):
        """Compute book sizes for all edges. Returns (max_book, count_max, sum_books, masked_matrix)."""
        M = A @ A
        masked = np.where(A > 0, M, 0)
        max_b = int(masked.max())
        count_max = int(np.sum(masked == max_b)) // 2
        sum_books = int(np.sum(masked)) // 2
        return max_b, count_max, sum_books, masked

    def cyclic_regular_init():
        """
        Create a 20-regular graph on 40 vertices using cyclic shifts.
        Vertex i connects to i+1, i+2, ..., i+10, i-1, i-2, ..., i-10 (mod 40).
        This gives exactly 40*20/2 = 400 edges.
        Then add 1 more edge to reach 401.
        """
        A = np.zeros((n, n), dtype=np.int8)
        # Each vertex connects to the next 10 and previous 10 (mod 40)
        for i in range(n):
            for offset in range(1, 11):
                j = (i + offset) % n
                k = (i - offset) % n
                A[i, j] = 1
                A[i, k] = 1
        # This gives 400 edges (each of 40 vertices has degree 20, total degree 800, edges = 400)
        # Add one more edge: pick two non-adjacent vertices
        # Find a pair (i,j) where A[i,j] == 0
        non_edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 0:
                    non_edges.append((i, j))
        if non_edges:
            # Pick a random non-edge
            u, v = random.choice(non_edges)
            A[u, v] = 1
            A[v, u] = 1
        return A

    def random_init():
        """Random graph with 401 edges."""
        all_edges = [(i, j) for i in range(n) for j in range(i+1, n)]
        random.shuffle(all_edges)
        A = np.zeros((n, n), dtype=np.int8)
        for u, v in all_edges[:target_edges]:
            A[u, v] = 1
            A[v, u] = 1
        return A

    def get_edges(A):
        """Return list of edges from adjacency matrix."""
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                if A[i, j] == 1:
                    edges.append((i, j))
        return edges

    def degree_preserving_swap(A, remove_edge, add_edge):
        """
        Try a degree-preserving swap: remove (u,v), add (w,x).
        For degree preservation, we need deg(u) and deg(v) to stay the same,
        and deg(w) and deg(x) to stay the same.
        This happens when: u is adjacent to x (so removing u-v and adding w-x keeps degs balanced)
        or v is adjacent to w, etc.
        
        Simpler approach: just do the swap and check if it's valid (all degrees preserved).
        """
        u, v = remove_edge
        w, x = add_edge
        
        # Check that the swap preserves degrees
        # Removing (u,v): deg(u)-1, deg(v)-1
        # Adding (w,x): deg(w)+1, deg(x)+1
        # For preservation: {u,v} and {w,x} must have the same degree multiset
        # OR: u==w or u==x (and v!=w and v!=x), etc.
        
        # Simplest valid degree-preserving swaps:
        # 1. u adjacent to x and v adjacent to w: swap (u,v) with (w,x) where u-x and v-w exist
        #    After: u loses v, gains x (if u-x didn't exist)... no wait.
        
        # Let me think more carefully. A degree-preserving swap means:
        # Before: edges include (u,v), not (w,x)
        # After: edges include (w,x), not (u,v)
        # deg changes: u: -1, v: -1, w: +1, x: +1
        # For all degrees to be preserved, we need the net change at each vertex to be 0.
        # This requires: {u,v} as a multiset equals {w,x} as a multiset, OR
        # one of u,v equals one of w,x, and the other pair also matches.
        
        # Case 1: u == w, v == x -> no change (skip)
        # Case 2: u == w, v != x -> deg(u) unchanged (loses v, gains x), deg(v): -1, deg(x): +1
        #   Need deg(v) == deg(x) for preservation? No, we just need the specific degrees to work out.
        #   Actually, for degree preservation we need: the set of vertices with degree change is empty.
        #   So we need: for each vertex, the number of removed incident edges equals added incident edges.
        
        # The cleanest degree-preserving swap: 
        # Remove (u,v) and (w,x), add (u,w) and (v,x) -- this is a 2-edge swap
        # But we can only remove/add one edge each to keep edge count constant.
        
        # For a single edge swap to preserve degrees:
        # We need deg(u)-1+I(u in {w,x}) = deg(u), so I(u in {w,x}) = 1
        # Similarly for v, w, x.
        # This means exactly one of {u,v} is in {w,x}, and exactly one of {w,x} is in {u,v}.
        # So: u==w and v!=x, or u==x and v!=w, etc.
        # If u==w: u loses v, gains x. deg(u) unchanged. v loses u: deg(v)-1. x gains u: deg(x)+1.
        #   Need deg(v) == deg(x) for this to preserve all degrees? No, we just need the final degrees to be valid.
        #   Actually, "degree-preserving" means the degree sequence is unchanged. So we need deg(v) == deg(x).
        
        # For simplicity, let's just do the swap and not worry about strict degree preservation.
        # The cyclic init gives us a good starting point, and small degree variations are OK.
        
        # Just perform the swap:
        A[u, v] = 0
        A[v, u] = 0
        A[w, x] = 1
        A[x, w] = 1

    def revert_swap(A, remove_edge, add_edge):
        """Revert a swap."""
        u, v = remove_edge
        w, x = add_edge
        A[u, v] = 1
        A[v, u] = 1
        A[w, x] = 0
        A[x, w] = 0

    best_score = None
    best_A = None
    
    # Try multiple initializations
    inits = [cyclic_regular_init, random_init, random_init, random_init]
    
    for init_func in inits:
        if time.time() > deadline - 20:
            break
            
        A = init_func()
        
        # Verify edge count
        edge_count = int(A.sum()) // 2
        if edge_count != target_edges:
            # Fix edge count
            edges = get_edges(A)
            if edge_count > target_edges:
                # Remove extra edges
                extra = edge_count - target_edges
                for _ in range(extra):
                    if not edges:
                        break
                    u, v = random.choice(edges)
                    A[u, v] = 0
                    A[v, u] = 0
                    edges.remove((u, v))
            elif edge_count < target_edges:
                # Add missing edges
                missing = target_edges - edge_count
                all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
                current_set = set(get_edges(A))
                non_edges = [e for e in all_possible if e not in current_set]
                random.shuffle(non_edges)
                for e in non_edges[:missing]:
                    u, v = e
                    A[u, v] = 1
                    A[v, u] = 1
        
        cur_max, cur_cnt, cur_sum, masked = calc_books(A)
        current_score = (cur_max, cur_cnt, cur_sum)
        
        if best_score is None or current_score < best_score:
            best_score = current_score
            best_A = A.copy()
        
        # Local search: targeted swaps to reduce max book
        stuck_counter = 0
        
        while time.time() < deadline - 5:
            cur_max, cur_cnt, cur_sum, masked = calc_books(A)
            current_score = (cur_max, cur_cnt, cur_sum)
            
            if current_score < best_score:
                best_score = current_score
                best_A = A.copy()
                stuck_counter = 0
                
                # Early termination if we hit the theoretical bound
                if best_score[0] <= 7:
                    break
            else:
                stuck_counter += 1
            
            # If stuck, do a random kick
            if stuck_counter > 50:
                edges = get_edges(A)
                all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
                current_set = set(edges)
                non_edges = [e for e in all_possible if e not in current_set]
                if edges and non_edges:
                    rem = random.choice(edges)
                    add = random.choice(non_edges)
                    A[rem[0], rem[1]] = 0
                    A[rem[1], rem[0]] = 0
                    A[add[0], add[1]] = 1
                    A[add[1], add[0]] = 1
                    stuck_counter = 0
                    continue
            
            # Find edges with maximum book size
            max_edges = []
            for i in range(n):
                for j in range(i+1, n):
                    if A[i, j] == 1 and masked[i, j] == cur_max:
                        max_edges.append((i, j))
            
            if not max_edges:
                break
            
            # Pick a random max-book edge to try to break
            u, v = random.choice(max_edges)
            
            # Try several swaps: remove (u,v), add a random non-edge
            best_swap = None
            best_swap_score = None
            best_add_edge = None
            
            all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
            current_set = set(get_edges(A))
            non_edges = [e for e in all_possible if e not in current_set and e != (u, v)]
            
            # Sample up to 20 candidate non-edges
            candidates = random.sample(non_edges, min(20, len(non_edges))) if non_edges else []
            
            for w, x in candidates:
                # Perform swap
                A[u, v] = 0
                A[v, u] = 0
                A[w, x] = 1
                A[x, w] = 1
                
                new_max, new_cnt, new_sum, _ = calc_books(A)
                new_score = (new_max, new_cnt, new_sum)
                
                if best_swap_score is None or new_score < best_swap_score:
                    best_swap_score = new_score
                    best_add_edge = (w, x)
                
                # Revert
                A[u, v] = 1
                A[v, u] = 1
                A[w, x] = 0
                A[x, w] = 0
            
            # Apply the best swap if it improves
            if best_add_edge is not None and best_swap_score < current_score:
                w, x = best_add_edge
                A[u, v] = 0
                A[v, u] = 0
                A[w, x] = 1
                A[x, w] = 1
                stuck_counter = 0

    # Final: extract edges from best_A
    edges = get_edges(best_A)
    
    # Ensure exactly 401 edges
    if len(edges) != target_edges:
        if len(edges) > target_edges:
            edges = edges[:target_edges]
        else:
            current_set = set(edges)
            all_possible = [(i, j) for i in range(n) for j in range(i+1, n)]
            non_edges = [e for e in all_possible if e not in current_set]
            random.shuffle(non_edges)
            for e in non_edges:
                if len(edges) == target_edges:
                    break
                edges.append(e)
    
    return [[u, v] for u, v in edges]
