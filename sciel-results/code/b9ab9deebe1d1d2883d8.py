import time
import random
import math
import numpy as np

def has_k33_fast(adj_bitmasks):
    """
    Check if graph is K3,3-free (no triple has >= 3 common neighbors).
    Returns True if valid.
    """
    n = len(adj_bitmasks)
    for i in range(n):
        ai = adj_bitmasks[i]
        for j in range(i + 1, n):
            common_ij = ai & adj_bitmasks[j]
            if common_ij.bit_count() < 3:
                continue
            for k in range(j + 1, n):
                common_ijk = common_ij & adj_bitmasks[k]
                if common_ijk.bit_count() >= 3:
                    return False
    return True

def is_add_valid(adj_bitmasks, u, v):
    """
    Check if adding edge (u,v) keeps the graph K3,3-free.
    """
    new_common_u = adj_bitmasks[v] & ~(1 << u)
    new_common_v = adj_bitmasks[u] & ~(1 << v)

    nc_u_list = []
    m = new_common_u
    while m:
        lsb = m & -m
        nc_u_list.append(lsb.bit_length() - 1)
        m -= lsb

    for ii in range(len(nc_u_list)):
        i = nc_u_list[ii]
        ai = adj_bitmasks[i]
        for jj in range(ii + 1, len(nc_u_list)):
            j = nc_u_list[jj]
            common = (adj_bitmasks[u] | (1 << v)) & ai & adj_bitmasks[j]
            if common.bit_count() >= 3:
                return False

    nc_v_list = []
    m = new_common_v
    while m:
        lsb = m & -m
        nc_v_list.append(lsb.bit_length() - 1)
        m -= lsb

    for ii in range(len(nc_v_list)):
        i = nc_u_list[ii] # Wait, logic error in original? No, nc_v_list is separate.
        # Actually, the original code has:
        # for ii in range(len(nc_v_list)):
        #     i = nc_v_list[ii]
        #     ...
        # Let's keep the original logic but ensure it's correct.
        i = nc_v_list[ii]
        ai = adj_bitmasks[i]
        for jj in range(ii + 1, len(nc_v_list)):
            j = nc_v_list[jj]
            common = (adj_bitmasks[v] | (1 << u)) & ai & adj_bitmasks[j]
            if common.bit_count() >= 3:
                return False

    return True

def construct():
    n = 20
    start_time = time.time()
    deadline = start_time + 115.0

    best_edges = 0
    best_adj_bitmasks = None

    def bitmasks_to_edges(bitmasks):
        edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    edges.append([i, j])
                m -= lsb
        return edges

    def count_edges(bitmasks):
        return sum(b.bit_count() for b in bitmasks) // 2

    def get_non_edges(bitmasks):
        non_edges = []
        for i in range(n):
            mi = bitmasks[i]
            for j in range(i + 1, n):
                if not (mi & (1 << j)):
                    non_edges.append((i, j))
        return non_edges

    def get_all_edges(bitmasks):
        all_edges = []
        for i in range(n):
            mask = bitmasks[i]
            m = mask
            while m:
                lsb = m & -m
                j = lsb.bit_length() - 1
                if i < j:
                    all_edges.append((i, j))
                m -= lsb
        return all_edges

    def greedy_add_exhaustive(bitmasks):
        changed = True
        while changed:
            changed = False
            non_edges = get_non_edges(bitmasks)
            random.shuffle(non_edges)
            for u, v in non_edges:
                if is_add_valid(bitmasks, u, v):
                    bitmasks[u] |= (1 << v)
                    bitmasks[v] |= (1 << u)
                    changed = True
        return bitmasks

    def local_search_sa(initial_bitmasks):
        """Local search with Simulated Annealing."""
        bitmasks = initial_bitmasks[:]
        current_edges = count_edges(bitmasks)
        best_local_edges = current_edges
        best_local_bitmasks = bitmasks[:]

        # Greedy start
        bitmasks = greedy_add_exhaustive(bitmasks)
        current_edges = count_edges(bitmasks)
        if current_edges > best_local_edges:
            best_local_edges = current_edges
            best_local_bitmasks = bitmasks[:]

        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling = 0.995
        steps = 0
        
        while time.time() < deadline:
            if T < T_min:
                break
            
            # Try a move
            move_type = random.choice(['add', 'remove', 'swap'])
            new_bitmasks = bitmasks[:]
            moved = False
            delta = 0

            if move_type == 'add':
                non_edges = get_non_edges(bitmasks)
                if non_edges:
                    u, v = random.choice(non_edges)
                    if is_add_valid(new_bitmasks, u, v):
                        new_bitmasks[u] |= (1 << v)
                        new_bitmasks[v] |= (1 << u)
                        moved = True
                        delta = 1 # Improvement in edges

            elif move_type == 'remove':
                all_edges = get_all_edges(bitmasks)
                if all_edges:
                    u, v = random.choice(all_edges)
                    new_bitmasks[u] &= ~(1 << v)
                    new_bitmasks[v] &= ~(1 << u)
                    moved = True
                    delta = -1 # Worsening

            elif move_type == 'swap':
                all_edges = get_all_edges(bitmasks)
                non_edges = get_non_edges(bitmasks)
                if all_edges and non_edges:
                    u, v = random.choice(all_edges)
                    x, y = random.choice(non_edges)
                    # Remove (u,v)
                    new_bitmasks[u] &= ~(1 << v)
                    new_bitmasks[v] &= ~(1 << u)
                    # Add (x,y)
                    if is_add_valid(new_bitmasks, x, y):
                        new_bitmasks[x] |= (1 << y)
                        new_bitmasks[y] |= (1 << x)
                        moved = True
                        delta = 0 # Same number of edges, but different structure
                    else:
                        # Revert removal to keep state valid if we don't accept?
                        # No, we only accept if valid. If invalid, we don't move.
                        moved = False

            if not moved:
                continue

            new_edges = current_edges + delta
            # Note: if delta is 0, we need to verify validity of the new state
            # is_add_valid ensures validity for the add part.
            # The remove part is always valid.
            # So if moved is True, new_bitmasks is valid.
            
            if new_edges > best_local_edges:
                best_local_edges = new_edges
                best_local_bitmasks = new_bitmasks[:]
                bitmasks = new_bitmasks
                current_edges = new_edges
                # Reset temperature? Or just keep cooling.
                # Keep cooling.
                
            else:
                # Accept worse or equal moves with probability
                # If delta > 0, always accept (handled above).
                # If delta <= 0:
                if delta > 0:
                    bitmasks = new_bitmasks
                    current_edges = new_edges
                else:
                    # delta is -1 or 0
                    if delta == 0:
                        # Accept with high prob to explore?
                        # Or use SA: exp(0/T) = 1. Always accept?
                        # If we always accept delta=0, we might cycle.
                        # Let's accept with prob 0.5 or use SA logic.
                        # SA: P = exp(-(-Delta)/T). Here Delta = E_new - E_old.
                        # We want to maximize edges.
                        # Energy E = -edges.
                        # Delta_E = E_new - E_old = -new_edges - (-current_edges) = current_edges - new_edges = -delta.
                        # P = exp(-Delta_E / T) = exp(delta / T).
                        # If delta = 0, P = 1.
                        # If delta = -1, P = exp(-1/T).
                        prob = math.exp(delta / T)
                        if random.random() < prob:
                            bitmasks = new_bitmasks
                            current_edges = new_edges
                            
            T *= cooling
            steps += 1

        return best_local_bitmasks, best_local_edges

    # Multiple restarts
    num_restarts = 100
    for r in range(num_restarts):
        if time.time() > deadline:
            break

        if r == 0:
            init_bitmasks = [0] * n
        elif r == 1:
            init_bitmasks = [0] * n
            p = 0.4
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
            attempts = 0
            while not has_k33_fast(init_bitmasks) and attempts < 3000:
                all_edges = get_all_edges(init_bitmasks)
                if not all_edges:
                    break
                u, v = random.choice(all_edges)
                init_bitmasks[u] &= ~(1 << v)
                init_bitmasks[v] &= ~(1 << u)
                attempts += 1
        else:
            init_bitmasks = [0] * n
            p = 0.2 + 0.02 * (r % 10)
            for i in range(n):
                for j in range(i + 1, n):
                    if random.random() < p:
                        init_bitmasks[i] |= (1 << j)
                        init_bitmasks[j] |= (1 << i)
            attempts = 0
            while not has_k33_fast(init_bitmasks) and attempts < 3000:
                all_edges = get_all_edges(init_bitmasks)
                if not all_edges:
                    break
                u, v = random.choice(all_edges)
                init_bitmasks[u] &= ~(1 << v)
                init_bitmasks[v] &= ~(1 << u)
                attempts += 1

        if not has_k33_fast(init_bitmasks):
            continue

        local_best_bitmasks, local_best_edges = local_search_sa(init_bitmasks)

        if local_best_edges > best_edges:
            best_edges = local_best_edges
            best_adj_bitmasks = local_best_bitmasks[:]

    if best_adj_bitmasks is not None:
        improved = greedy_add_exhaustive(best_adj_bitmasks[:])
        improved_edges = count_edges(improved)
        if improved_edges > best_edges:
            best_edges = improved_edges
            best_adj_bitmasks = improved

    if best_adj_bitmasks is None:
        return []

    return bitmasks_to_edges(best_adj_bitmasks)

if __name__ == "__main__":
    t0 = time.time()
    edges = construct()
    t1 = time.time()
    print(f"Found {len(edges)} edges in {t1-t0:.2f}s")
    n = 20
    adj = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        adj[u, v] = True
        adj[v, u] = True
    valid = True
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                common = adj[i] & adj[j] & adj[k]
                if np.sum(common) >= 3:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    print(f"Valid K3,3-free: {valid}")
