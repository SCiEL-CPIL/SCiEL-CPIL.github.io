"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools

N = 20
ALL_EDGES = [(i, j) for i in range(N) for j in range(i+1, N)]
ALL_EDGES_SET = set(ALL_EDGES)

def is_valid_add(u, v, adj):
    """
    Check if adding edge (u, v) to graph represented by adj (list of bitmasks)
    results in a K3,3-free graph.
    
    When adding edge (u,v), a new K3,3 must use this edge. WLOG u in part A, v in part B.
    We need: two more vertices a1,a2 in A (adjacent to v) and two more b1,b2 in B (adjacent to u)
    such that all cross edges exist.
    
    Equivalent: for pairs (x,y) both adjacent to u, check if |N(x) ∩ N(y) ∩ N(v)| >= 2.
    """
    nu = adj[u]
    nv = adj[v]
    
    # Get neighbors of u (excluding u and v)
    x_list = []
    for x in range(N):
        if x != u and x != v and (nu >> x) & 1:
            x_list.append(x)
    
    # Check all pairs in x_list
    for i in range(len(x_list)):
        x = x_list[i]
        ax = adj[x]
        for j in range(i+1, len(x_list)):
            y = x_list[j]
            common = ax & adj[y] & nv
            if common.bit_count() >= 2:
                return False
    
    return True

def add_edge(u, v, adj):
    adj[u] |= (1 << v)
    adj[v] |= (1 << u)

def remove_edge(u, v, adj):
    adj[u] &= ~(1 << v)
    adj[v] &= ~(1 << u)

def count_edges(adj):
    return sum(adj[i].bit_count() for i in range(N)) // 2

def get_missing_edges(adj):
    missing = []
    for u, v in ALL_EDGES:
        if not ((adj[u] >> v) & 1):
            missing.append((u, v))
    return missing

def get_current_edges(adj):
    current = []
    for u, v in ALL_EDGES:
        if (adj[u] >> v) & 1:
            current.append((u, v))
    return current

def try_add_edges(adj, edges_to_try, max_add, deadline):
    """Try to add up to max_add edges from edges_to_try to adj."""
    added = []
    for u, v in edges_to_try:
        if time.time() > deadline:
            break
        if len(added) >= max_add:
            break
        if not ((adj[u] >> v) & 1):
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
                added.append((u, v))
    return added

def hill_climbing(adj, deadline):
    """
    Hill climbing with add, 1-swap (remove 1 add 2), and 2-swap (remove 2 add 3).
    """
    improved = True
    no_improve_rounds = 0
    max_no_improve = 3
    
    while improved and time.time() < deadline:
        improved = False
        
        # Phase 1: Try to add any missing edge
        missing = get_missing_edges(adj)
        random.shuffle(missing)
        for u, v in missing:
            if time.time() > deadline:
                return adj
            if is_valid_add(u, v, adj):
                add_edge(u, v, adj)
                improved = True
                break
        
        if time.time() > deadline:
            return adj
        
        if improved:
            no_improve_rounds = 0
            continue
        
        no_improve_rounds += 1
        if no_improve_rounds > max_no_improve:
            break
        
        # Phase 2: Try 1-swap (remove 1 edge, add 2 edges)
        current = get_current_edges(adj)
        random.shuffle(current)
        found = False
        for u1, v1 in current:
            if time.time() > deadline:
                return adj
            
            remove_edge(u1, v1, adj)
            
            missing = get_missing_edges(adj)
            random.shuffle(missing)
            added = try_add_edges(adj, missing, 2, deadline)
            
            if len(added) == 2:
                improved = True
                found = True
                break
            else:
                for u2, v2 in added:
                    remove_edge(u2, v2, adj)
                add_edge(u1, v1, adj)
        
        if time.time() > deadline:
            return adj
        if found:
            no_improve_rounds = 0
            continue
        
        # Phase 3: Try 2-swap (remove 2 edges, add 3 edges)
        current = get_current_edges(adj)
        random.shuffle(current)
        found = False
        n_cur = len(current)
        for i in range(n_cur):
            if time.time() > deadline:
                return adj
            u1, v1 = current[i]
            for j in range(i+1, n_cur):
                if time.time() > deadline:
                    return adj
                u2, v2 = current[j]
                
                remove_edge(u1, v1, adj)
                remove_edge(u2, v2, adj)
                
                missing = get_missing_edges(adj)
                random.shuffle(missing)
                added = try_add_edges(adj, missing, 3, deadline)
                
                if len(added) == 3:
                    improved = True
                    found = True
                    break
                else:
                    for u3, v3 in added:
                        remove_edge(u3, v3, adj)
                    add_edge(u1, v1, adj)
                    add_edge(u2, v2, adj)
            
            if found:
                break
        
        if time.time() > deadline:
            return adj
        if found:
            no_improve_rounds = 0
            continue
    
    return adj

def greedy_construct(adj, deadline, seed=None):
    """Greedy construction with random edge order."""
    if seed is not None:
        random.seed(seed)
    
    shuffled_edges = ALL_EDGES[:]
    random.shuffle(shuffled_edges)
    
    for u, v in shuffled_edges:
        if time.time() > deadline:
            break
        if is_valid_add(u, v, adj):
            add_edge(u, v, adj)
    
    return adj

def simulated_annealing_step(adj, deadline, temperature):
    """
    A single simulated annealing step: try a random move (add, remove, or swap)
    and accept based on the Metropolis criterion.
    """
    move_type = random.random()
    
    if move_type < 0.4:
        # Try to add a random missing edge
        missing = get_missing_edges(adj)
        if not missing:
            return adj, 0
        u, v = random.choice(missing)
        if is_valid_add(u, v, adj):
            add_edge(u, v, adj)
            return adj, 1
        return adj, 0
    
    elif move_type < 0.7:
        # Try a 1-swap: remove 1 random edge, add up to 2
        current = get_current_edges(adj)
        if not current:
            return adj, 0
        u1, v1 = random.choice(current)
        
        remove_edge(u1, v1, adj)
        missing = get_missing_edges(adj)
        random.shuffle(missing)
        added = try_add_edges(adj, missing, 2, deadline)
        
        if len(added) == 2:
            return adj, 1
        else:
            for u2, v2 in added:
                remove_edge(u2, v2, adj)
            add_edge(u1, v1, adj)
            return adj, 0
    
    else:
        # Try a 2-swap: remove 2 random edges, add up to 3
        current = get_current_edges(adj)
        if len(current) < 2:
            return adj, 0
        idx1 = random.randint(0, len(current)-1)
        idx2 = random.randint(0, len(current)-1)
        if idx1 == idx2:
            idx2 = (idx2 + 1) % len(current)
        u1, v1 = current[idx1]
        u2, v2 = current[idx2]
        
        remove_edge(u1, v1, adj)
        remove_edge(u2, v2, adj)
        
        missing = get_missing_edges(adj)
        random.shuffle(missing)
        added = try_add_edges(adj, missing, 3, deadline)
        
        if len(added) == 3:
            return adj, 1
        else:
            for u3, v3 in added:
                remove_edge(u3, v3, adj)
            add_edge(u1, v1, adj)
            add_edge(u2, v2, adj)
            return adj, 0

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin (evaluator kills at 150s)
    
    best_adj = None
    best_count = 0
    
    # Try multiple random restarts with different strategies
    restart = 0
    while time.time() < deadline - 10:
        restart += 1
        seed = restart * 12345 + 67890
        
        # Greedy construction with random edge order
        adj = [0] * N
        adj = greedy_construct(adj, deadline - 10, seed)
        
        # Hill climbing
        adj = hill_climbing(adj, deadline - 10)
        
        current_count = count_edges(adj)
        if current_count > best_count:
            best_count = current_count
            best_adj = adj[:]
        
        # If we have time, do some simulated annealing
        if time.time() < deadline - 15 and best_count > 0:
            sa_adj = best_adj[:]
            temperature = 0.5
            for step in range(50):
                if time.time() > deadline - 10:
                    break
                sa_adj, improvement = simulated_annealing_step(sa_adj, deadline - 10, temperature)
                sa_count = count_edges(sa_adj)
                if sa_count > best_count:
                    best_count = sa_count
                    best_adj = sa_adj[:]
                else:
                    # Revert if worse
                    sa_adj = best_adj[:]
                temperature *= 0.95
        
        # Early stop if we hit a high threshold
        if best_count >= 80:
            break

    if best_adj is None:
        return []
        
    # Convert adj back to list of edges
    edges = []
    for u in range(N):
        for v in range(u+1, N):
            if (best_adj[u] >> v) & 1:
                edges.append([u, v])
                
    return edges

# EVOLVE-BLOCK-END
