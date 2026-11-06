import time
import random
import math

N = 20
C = 20
DEADLINE = 145.0


def _validate(edges):
    """Return True iff every 4-cycle is rainbow."""
    if not edges:
        return True
    adj = {}
    neigh = [set() for _ in range(N)]
    for u, v, c in edges:
        adj[(u, v)] = c
        adj[(v, u)] = c
        neigh[u].add(v)
        neigh[v].add(u)

    for u in range(N):
        for v in neigh[u]:
            if u >= v:
                continue
            common = neigh[u] & neigh[v]
            common.discard(u)
            common.discard(v)
            common_list = list(common)
            for i in range(len(common_list)):
                x = common_list[i]
                for j in range(i + 1, len(common_list)):
                    y = common_list[j]
                    cols = {adj[(u, x)], adj[(x, v)], adj[(v, y)], adj[(y, u)]}
                    if len(cols) != 4:
                        return False
    return True


def _check_add(adj, neigh, u, v, c):
    """Check if adding edge (u,v) with color c maintains rainbow 4-cycles."""
    # 4-cycles: u-v-x-y-u where x~v, y~u, x!=y, x~y
    for x in neigh[v]:
        if x == u:
            continue
        for y in neigh[u]:
            if y == v or y == x:
                continue
            if y not in neigh[x]:
                continue
            # cycle u-v-x-y-u
            # colors: c, adj[(v,x)], adj[(x,y)], adj[(y,u)]
            if c == adj[(v, x)] or c == adj[(x, y)] or c == adj[(y, u)]:
                return False
            if adj[(v, x)] == adj[(x, y)] or adj[(v, x)] == adj[(y, u)]:
                return False
            if adj[(x, y)] == adj[(y, u)]:
                return False
    return True


class GraphState:
    """Efficient graph state for tracking edges, colors, and neighborhoods."""
    def __init__(self):
        self.adj = {}  # (u,v) -> color
        self.neigh = [set() for _ in range(N)]
        self.color_at = [[None]*N for _ in range(N)]
        self.edge_set = set()  # frozenset of (min(u,v), max(u,v))
        self.edges = []  # list of [u,v,c]

    def add_edge(self, u, v, c):
        self.adj[(u, v)] = c
        self.adj[(v, u)] = c
        self.neigh[u].add(v)
        self.neigh[v].add(u)
        self.color_at[u][v] = c
        self.color_at[v][u] = c
        key = (min(u,v), max(u,v))
        self.edge_set.add(key)
        self.edges.append([u, v, c])

    def remove_edge(self, u, v):
        c = self.adj.pop((u, v))
        self.adj.pop((v, u))
        self.neigh[u].discard(v)
        self.neigh[v].discard(u)
        self.color_at[u][v] = None
        self.color_at[v][u] = None
        key = (min(u,v), max(u,v))
        self.edge_set.discard(key)
        # Remove from edges list
        for i, (a, b, cc) in enumerate(self.edges):
            if (a == u and b == v) or (a == v and b == u):
                self.edges.pop(i)
                break

    def available_colors(self, u, v):
        used_u = set(self.color_at[u][w] for w in self.neigh[u] if self.color_at[u][w] is not None)
        used_v = set(self.color_at[v][w] for w in self.neigh[v] if self.color_at[v][w] is not None)
        return [c for c in range(C) if c not in used_u and c not in used_v]

    def try_add(self, u, v, rng):
        """Try to add edge (u,v) with a valid color. Returns True if added."""
        if (min(u,v), max(u,v)) in self.edge_set:
            return False
        available = self.available_colors(u, v)
        if not available:
            return False
        rng.shuffle(available)
        for c in available:
            if _check_add(self.adj, self.neigh, u, v, c):
                self.add_edge(u, v, c)
                return True
        return False

    def try_recolor(self, u, v, rng):
        """Try to recolor edge (u,v). Returns True if recolored."""
        if (min(u,v), max(u,v)) not in self.edge_set:
            return False
        old_c = self.adj[(u, v)]
        # Remove edge
        self.remove_edge(u, v)
        # Try to add back with different color
        available = self.available_colors(u, v)
        if old_c in available:
            available.remove(old_c)
        if not available:
            # Put it back
            self.add_edge(u, v, old_c)
            return False
        rng.shuffle(available)
        for c in available:
            if _check_add(self.adj, self.neigh, u, v, c):
                self.add_edge(u, v, c)
                return True
        # Put it back with original color
        self.add_edge(u, v, old_c)
        return False


def _greedy_build(order, rng, time_budget):
    """Build graph greedily in given edge order."""
    state = GraphState()
    for (u, v) in order:
        if time.time() > time_budget:
            break
        state.try_add(u, v, rng)
    return state


def _simulated_annealing(state, rng, time_budget):
    """Simulated annealing to improve edge count."""
    current_edges = len(state.edges)
    best_state = state
    best_edges = current_edges
    T = 1.0
    cooling = 0.9995
    
    while time.time() < time_budget:
        T *= cooling
        if T < 0.01:
            break
            
        # Random move: add, remove, or recolor
        move_type = rng.random()
        if move_type < 0.4:
            # Try to add a random missing edge
            all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
            missing = [(u,v) for (u,v) in all_pairs if (u,v) not in state.edge_set]
            if not missing:
                continue
            u, v = rng.choice(missing)
            if state.try_add(u, v, rng):
                new_edges = len(state.edges)
                if new_edges > best_edges:
                    best_edges = new_edges
            else:
                # Accept with probability exp(-1/T)
                if rng.random() < math.exp(-1.0 / T):
                    # Accept anyway (no change, but keep going)
                    pass
        elif move_type < 0.7:
            # Try to remove a random edge then re-add
            if not state.edges:
                continue
            u, v, c = rng.choice(state.edges)
            state.remove_edge(u, v)
            # Try to add it back
            if not state.try_add(u, v, rng):
                # Try to add other edges
                for _ in range(3):
                    all_pairs = [(a, b) for a in range(N) for b in range(a+1, N)]
                    missing = [(a,b) for (a,b) in all_pairs if (a,b) not in state.edge_set]
                    if missing:
                        a, b = rng.choice(missing)
                        state.try_add(a, b, rng)
            new_edges = len(state.edges)
            if new_edges > best_edges:
                best_edges = new_edges
        else:
            # Recolor a random edge
            if not state.edges:
                continue
            u, v, c = rng.choice(state.edges)
            state.try_recolor(u, v, rng)
            new_edges = len(state.edges)
            if new_edges > best_edges:
                best_edges = new_edges
    
    return state, best_edges


def _local_search_add(state, rng, time_budget):
    """Local search: repeatedly try to add missing edges."""
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    improved = True
    while improved and time.time() < time_budget:
        improved = False
        missing = [(u,v) for (u,v) in all_pairs if (u,v) not in state.edge_set]
        if not missing:
            break
        rng.shuffle(missing)
        for (u, v) in missing:
            if time.time() > time_budget:
                break
            if state.try_add(u, v, rng):
                improved = True
    return state


def _remove_and_refill(state, rng, time_budget):
    """Remove a random edge, then try to add multiple edges back."""
    if not state.edges:
        return state
    u, v, c = rng.choice(state.edges)
    state.remove_edge(u, v)
    
    all_pairs = [(a, b) for a in range(N) for b in range(a+1, N)]
    missing = [(a,b) for (a,b) in all_pairs if (a,b) not in state.edge_set]
    rng.shuffle(missing)
    for (a, b) in missing[:20]:
        if time.time() > time_budget:
            break
        state.try_add(a, b, rng)
    return state


def construct():
    start = time.time()
    best_state = None
    best_len = 0
    
    all_pairs = [(u, v) for u in range(N) for v in range(u+1, N)]
    
    # Multiple random greedy attempts with different strategies
    for trial in range(40):
        if time.time() > DEADLINE - 30:
            break
        
        rng = random.Random(trial * 137 + 42)
        
        if trial < 15:
            # Random edge order
            order = all_pairs[:]
            rng.shuffle(order)
            state = _greedy_build(order, rng, DEADLINE - 30)
        elif trial < 25:
            # Structured: vertex permutation based ordering
            perm = list(range(N))
            rng.shuffle(perm)
            order = []
            for i in range(N):
                for j in range(i+1, N):
                    order.append((perm[i], perm[j]))
            rng.shuffle(order)
            state = _greedy_build(order, rng, DEADLINE - 30)
        elif trial < 35:
            # Low colors first
            order = all_pairs[:]
            rng.shuffle(order)
            state = GraphState()
            for (u, v) in order:
                if time.time() > DEADLINE - 30:
                    break
                available = state.available_colors(u, v)
                if not available:
                    continue
                for c in sorted(available):
                    if _check_add(state.adj, state.neigh, u, v, c):
                        state.add_edge(u, v, c)
                        break
        else:
            # High colors first
            order = all_pairs[:]
            rng.shuffle(order)
            state = GraphState()
            for (u, v) in order:
                if time.time() > DEADLINE - 30:
                    break
                available = state.available_colors(u, v)
                if not available:
                    continue
                for c in sorted(available, reverse=True):
                    if _check_add(state.adj, state.neigh, u, v, c):
                        state.add_edge(u, v, c)
                        break
        
        if len(state.edges) > best_len:
            best_len = len(state.edges)
            best_state = state
    
    # Local search on best
    if best_state and time.time() < DEADLINE - 20:
        rng = random.Random(12345)
        _local_search_add(best_state, rng, DEADLINE - 20)
        if len(best_state.edges) > best_len:
            best_len = len(best_state.edges)
    
    # Simulated annealing
    if best_state and time.time() < DEADLINE - 10:
        rng = random.Random(99999)
        best_state, sa_best = _simulated_annealing(best_state, rng, DEADLINE - 10)
        if sa_best > best_len:
            best_len = sa_best
    
    # Final local search
    if best_state and time.time() < DEADLINE - 3:
        rng = random.Random(77777)
        _local_search_add(best_state, rng, DEADLINE - 3)
        if len(best_state.edges) > best_len:
            best_len = len(best_state.edges)
    
    # Remove-and-refill attempts
    if best_state and time.time() < DEADLINE - 2:
        rng = random.Random(55555)
        for _ in range(5):
            if time.time() > DEADLINE - 2:
                break
            _remove_and_refill(best_state, rng, DEADLINE - 2)
            if len(best_state.edges) > best_len:
                best_len = len(best_state.edges)
    
    if not best_state or not best_state.edges:
        # Fallback: star
        return [[0, v, v] for v in range(1, N)]
    
    return best_state.edges


if __name__ == "__main__":
    res = construct()
    print("edges:", len(res))
    print("valid:", _validate(res))
