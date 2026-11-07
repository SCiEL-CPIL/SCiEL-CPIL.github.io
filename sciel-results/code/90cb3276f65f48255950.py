"""Combined approach: Start from K_{a,b} bipartite (structurally valid when a!=b),
then use fast incremental local search with 2-swap moves to maximize edges."""
# EVOLVE-BLOCK-START
import time
import random
from collections import defaultdict

N = 15
DEADLINE = 85.0  # Safety margin before 90s kill

class _GraphState:
    """Maintains incremental graph state for fast validity checks."""
    
    def __init__(self, adj=None):
        self.adj = [[0] * N for _ in range(N)]
        self.deg = [0] * N
        self.neighbors = [set() for _ in range(N)]
        if adj is not None:
            self._load(adj)
    
    def _load(self, adj):
        for u in range(N):
            for v in range(u + 1, N):
                if adj[u][v]:
                    self.add_edge(u, v)
    
    def add_edge(self, u, v):
        if u == v or self.adj[u][v]:
            return False
        self.adj[u][v] = 1
        self.adj[v][u] = 1
        self.deg[u] += 1
        self.deg[v] += 1
        self.neighbors[u].add(v)
        self.neighbors[v].add(u)
        return True
    
    def remove_edge(self, u, v):
        if u == v or not self.adj[u][v]:
            return False
        self.adj[u][v] = 0
        self.adj[v][u] = 0
        self.deg[u] -= 1
        self.deg[v] -= 1
        self.neighbors[u].discard(v)
        self.neighbors[v].discard(u)
        return True
    
    def edge_count(self):
        return sum(self.deg) // 2
    
    def copy(self):
        new_state = _GraphState()
        for u in range(N):
            new_state.adj[u] = self.adj[u][:]
            new_state.deg[u] = self.deg[u]
            new_state.neighbors[u] = set(self.neighbors[u])
        return new_state

def _is_valid_state(state):
    """
    Check if graph is valid: no simple 3-edge path s-u-v-t has deg(s) == deg(t).
    For each edge (u,v): no degree appears in both N(u)\{v} and N(v)\{u}
    (with the caveat that if the only vertex with that degree on both sides is the same vertex, it's fine).
    """
    deg = state.deg
    neighbors = state.neighbors
    
    for u in range(N):
        for v in neighbors[u]:
            if u >= v:
                continue
            
            # Build mapping: degree -> set of vertices in N(u)\{v}
            deg_map_u = defaultdict(set)
            for s in neighbors[u]:
                if s != v:
                    d = deg[s]
                    deg_map_u[d].add(s)
            
            # Check each t in N(v)\{u}
            for t in neighbors[v]:
                if t == u:
                    continue
                d = deg[t]
                if d in deg_map_u:
                    # Check if there exists s in deg_map_u[d] with s != t
                    if len(deg_map_u[d]) > 1 or (t not in deg_map_u[d]):
                        return False
    return True

def _try_add_state(state, u, v):
    if u == v or state.adj[u][v]:
        return False
    state.add_edge(u, v)
    if _is_valid_state(state):
        return True
    state.remove_edge(u, v)
    return False

def _try_swap_state(state, u1, v1, u2, v2):
    if state.adj[u1][v1] == 0 or state.adj[u2][v2] == 1:
        return False
    state.remove_edge(u1, v1)
    state.add_edge(u2, v2)
    if _is_valid_state(state):
        return True
    state.remove_edge(u2, v2)
    state.add_edge(u1, v1)
    return False

def _try_2swap_state(state, u1, v1, u2, v2, u3, v3, u4, v4):
    """Remove edges (u1,v1) and (u2,v2), add edges (u3,v3) and (u4,v4)."""
    if (state.adj[u1][v1] == 0 or state.adj[u2][v2] == 0 or 
        state.adj[u3][v3] == 1 or state.adj[u4][v4] == 1):
        return False
    # Check no duplicate edges in the swap
    if (u1 == u3 and v1 == v3) or (u1 == u4 and v1 == v4) or \
       (u2 == u3 and v2 == v3) or (u2 == u4 and v2 == v4):
        return False
    
    state.remove_edge(u1, v1)
    state.remove_edge(u2, v2)
    state.add_edge(u3, v3)
    state.add_edge(u4, v4)
    if _is_valid_state(state):
        return True
    state.remove_edge(u4, v4)
    state.remove_edge(u3, v3)
    state.add_edge(u2, v2)
    state.add_edge(u1, v1)
    return False

def _local_search(state, start_time, seed):
    """Hill climbing with add, swap, and 2-swap moves."""
    random.seed(seed)
    current_count = state.edge_count()
    best_state = state.copy()
    best_count = current_count
    no_improve_rounds = 0
    
    while time.time() - start_time < DEADLINE:
        improved = False
        
        # Try adding missing edges
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not state.adj[u][v]]
        random.shuffle(missing)
        for u, v in missing:
            if time.time() - start_time > DEADLINE:
                break
            if _try_add_state(state, u, v):
                current_count += 1
                improved = True
                if current_count > best_count:
                    best_count = current_count
                    best_state = state.copy()
        
        if improved:
            no_improve_rounds = 0
            continue
        
        no_improve_rounds += 1
        if no_improve_rounds > 3:
            # Try 2-swaps before giving up
            if no_improve_rounds > 8:
                break
            
            # Try 2-swaps: remove 2 existing edges, add 2 missing edges
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) if state.adj[u][v]]
            missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not state.adj[u][v]]
            random.shuffle(existing)
            random.shuffle(missing)
            
            found_2swap = False
            max_existing = min(len(existing), 30)
            max_missing = min(len(missing), 30)
            for i in range(max_existing):
                if time.time() - start_time > DEADLINE:
                    break
                u1, v1 = existing[i]
                for j in range(i + 1, max_existing):
                    u2, v2 = existing[j]
                    for k in range(max_missing):
                        if time.time() - start_time > DEADLINE:
                            break
                        u3, v3 = missing[k]
                        for l in range(k + 1, max_missing):
                            u4, v4 = missing[l]
                            if _try_2swap_state(state, u1, v1, u2, v2, u3, v3, u4, v4):
                                found_2swap = True
                                break
                        if found_2swap:
                            break
                    if found_2swap:
                        break
                if found_2swap:
                    break
            
            if found_2swap:
                no_improve_rounds = 0
                continue
            
            # Fall back to single swaps
            existing = [(u, v) for u in range(N) for v in range(u + 1, N) if state.adj[u][v]]
            missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not state.adj[u][v]]
            random.shuffle(existing)
            random.shuffle(missing)
            
            found_swap = False
            for u1, v1 in existing:
                if time.time() - start_time > DEADLINE:
                    break
                for u2, v2 in missing:
                    if _try_swap_state(state, u1, v1, u2, v2):
                        found_swap = True
                        break
                if found_swap:
                    break
            
            if found_swap:
                no_improve_rounds = 0
                continue
            
            continue
        
        # Try swaps: remove one existing edge, add one missing edge
        existing = [(u, v) for u in range(N) for v in range(u + 1, N) if state.adj[u][v]]
        missing = [(u, v) for u in range(N) for v in range(u + 1, N) if not state.adj[u][v]]
        random.shuffle(existing)
        random.shuffle(missing)
        
        found_swap = False
        for u1, v1 in existing:
            if time.time() - start_time > DEADLINE:
                break
            for u2, v2 in missing:
                if _try_swap_state(state, u1, v1, u2, v2):
                    found_swap = True
                    break
            if found_swap:
                break
        
        if not found_swap:
            break
    
    return best_state, best_count

def _bipartite_start(a_size):
    """Create K_{a_size, N-a_size}."""
    adj = [[0] * N for _ in range(N)]
    for u in range(a_size):
        for v in range(a_size, N):
            adj[u][v] = 1
            adj[v][u] = 1
    return adj

def construct():
    start_time = time.time()
    best_state = _GraphState()
    best_count = 0
    
    # Try different bipartite splits as starting points
    for a_size in [7, 8, 6, 9, 5, 10]:
        if time.time() - start_time > DEADLINE - 5:
            break
        adj = _bipartite_start(a_size)
        state = _GraphState(adj)
        count = state.edge_count()
        
        # Local search from this starting point
        state_ls, count_ls = _local_search(state, start_time, seed=a_size * 100)
        if count_ls > best_count:
            best_count = count_ls
            best_state = state_ls
    
    # Also try a few random greedy starts
    for i in range(10):
        if time.time() - start_time > DEADLINE - 5:
            break
        random.seed(10000 + i)
        adj = [[0] * N for _ in range(N)]
        edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(edges)
        state = _GraphState()
        for u, v in edges:
            if time.time() - start_time > DEADLINE - 3:
                break
            _try_add_state(state, u, v)
        count = state.edge_count()
        if count > best_count:
            # Do local search from this point
            state_ls, count_ls = _local_search(state, start_time, seed=20000 + i)
            if count_ls > best_count:
                best_count = count_ls
                best_state = state_ls
    
    # Final validation
    if not _is_valid_state(best_state):
        # Fallback to K_{7,8}
        adj = _bipartite_start(7)
        best_state = _GraphState(adj)
        best_count = best_state.edge_count()
    
    result = []
    for u in range(N):
        for v in range(u + 1, N):
            if best_state.adj[u][v]:
                result.append([int(u), int(v)])
    
    return result

# EVOLVE-BLOCK-END
