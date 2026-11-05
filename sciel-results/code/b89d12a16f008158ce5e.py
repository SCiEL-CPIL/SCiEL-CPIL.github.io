"""Rewritten program: maximize edges of a 20-vertex graph, 20 colours, every 4-cycle rainbow."""
import random
import time

N = 20
K = 20
DEADLINE = 110.0  # safety margin under 120s wall-clock

# Precompute common neighbours for fast validity check
_COMMON = [[[] for _ in range(N)] for _ in range(N)]
for u in range(N):
    for v in range(u + 1, N):
        lst = [w for w in range(N) if w != u and w != v]
        _COMMON[u][v] = lst
        _COMMON[v][u] = lst

# Precompute all pairs
_PAIRS = [(u, v) for u in range(N) for v in range(u + 1, N)]

class State:
    __slots__ = ('edge', 'color_at', 'color_count', 'adj')
    # edge: dict (u,v) -> c
    # color_at: list of sets of colors used at each vertex
    # color_count: list of counts for each color
    # adj: list of sets of neighbors

    def __init__(self):
        self.edge = {}
        self.color_at = [set() for _ in range(N)]
        self.color_count = [0] * K
        self.adj = [set() for _ in range(N)]

    def add(self, u, v, c):
        key = (min(u, v), max(u, v))
        self.edge[key] = c
        self.color_at[u].add(c)
        self.color_at[v].add(c)
        self.color_count[c] += 1
        self.adj[u].add(v)
        self.adj[v].add(u)

    def remove(self, u, v):
        key = (min(u, v), max(u, v))
        if key not in self.edge:
            return None
        c = self.edge.pop(key)
        self.color_at[u].discard(c)
        self.color_at[v].discard(c)
        self.color_count[c] -= 1
        self.adj[u].discard(v)
        self.adj[v].discard(u)
        return c

    def recolor(self, u, v, new_c):
        key = (min(u, v), max(u, v))
        if key not in self.edge:
            return False
        old_c = self.edge[key]
        if old_c == new_c:
            return True
        # Check if new_c is valid for u, v
        if not self.valid_add(u, v, new_c):
            return False
        # Remove old
        self.color_at[u].discard(old_c)
        self.color_at[v].discard(old_c)
        self.color_count[old_c] -= 1
        # Add new
        self.edge[key] = new_c
        self.color_at[u].add(new_c)
        self.color_at[v].add(new_c)
        self.color_count[new_c] += 1
        return True

    def valid_add(self, u, v, c):
        # Check if c is already used at u or v
        if c in self.color_at[u] or c in self.color_at[v]:
            return False
        # Check if any common neighbor w has color c
        for w in _COMMON[u][v]:
            if c in self.color_at[w]:
                return False
        return True

    def count(self):
        return len(self.edge)

def greedy_build(rng):
    st = State()
    # Use a heuristic order: vertices with higher potential degree first? 
    # For now, shuffle pairs but pick colors intelligently.
    pairs = _PAIRS[:]
    rng.shuffle(pairs)
    
    for u, v in pairs:
        if u in st.adj[v]: # already connected
            continue
        # Find best color: one that is not used at u, v, or any common neighbor
        # and has the lowest global count to balance usage
        best_c = -1
        min_count = float('inf')
        for c in range(K):
            if c in st.color_at[u] or c in st.color_at[v]:
                continue
            # Check common neighbors
            valid = True
            for w in _COMMON[u][v]:
                if c in st.color_at[w]:
                    valid = False
                    break
            if not valid:
                continue
            if st.color_count[c] < min_count:
                min_count = st.color_count[c]
                best_c = c
        if best_c != -1:
            st.add(u, v, best_c)
    return st

def local_search(st, rng, time_limit):
    """Try to add edges by recoloring existing edges."""
    improved = True
    start_time = time.time()
    while improved and time.time() < time_limit:
        improved = False
        # Try to add missing edges
        missing = [(u, v) for u, v in _PAIRS if u not in st.adj[v]]
        rng.shuffle(missing)
        
        for u, v in missing:
            if time.time() > time_limit:
                break
            # Find a color c that is blocked by a common neighbor w
            # i.e., c not in color_at[u], c not in color_at[v], but c in color_at[w] for some w in common(u,v)
            # We want to recolor an edge incident to w to free up c.
            
            # First, find candidate colors that are "almost" valid
            candidates = []
            for c in range(K):
                if c in st.color_at[u] or c in st.color_at[v]:
                    continue
                # Check if it's blocked
                blocked_by = []
                for w in _COMMON[u][v]:
                    if c in st.color_at[w]:
                        blocked_by.append(w)
                if blocked_by:
                    candidates.append((c, blocked_by))
            
            if not candidates:
                # No candidates means either no color available or all are fully blocked by multiple constraints
                # If a color is valid, we would have added it in greedy. 
                # So if no candidates, it means for all c not in u/v, there is at least one w with c.
                # We need to recolor.
                pass
            
            # Try to resolve one candidate
            for c, blockers in candidates:
                # Pick a random blocker w
                w = rng.choice(blockers)
                # Find an edge (w, x) with color c
                # We need to recolor (w, x) to a new color d
                # such that d is valid for w, x and d is not c (obviously)
                # and ideally d doesn't block other things.
                
                # Find neighbors of w
                neighbors_w = list(st.adj[w])
                rng.shuffle(neighbors_w)
                
                found_recolor = False
                for x in neighbors_w:
                    key_wx = (min(w, x), max(w, x))
                    if st.edge.get(key_wx) != c:
                        continue
                    
                    # Try to find a new color d for (w, x)
                    for d in range(K):
                        if d == c:
                            continue
                        if d in st.color_at[w] or d in st.color_at[x]:
                            continue
                        # Check common neighbors of w, x for d
                        valid_d = True
                        for z in _COMMON[w][x]:
                            if d in st.color_at[z]:
                                valid_d = False
                                break
                        if not valid_d:
                            continue
                        
                        # Perform recolor
                        # Temporarily remove c from w, x? No, recolor handles it.
                        # But we must ensure that after recoloring (w,x) to d,
                        # the color c is no longer present at w.
                        # If (w,x) was the ONLY edge of color c at w, then c is freed at w.
                        # If there are other edges of color c at w, c is still blocked.
                        
                        # Check if c is used elsewhere at w
                        c_used_elsewhere = False
                        for nb in st.adj[w]:
                            if nb == x:
                                continue
                            key_w_nb = (min(w, nb), max(w, nb))
                            if st.edge.get(key_w_nb) == c:
                                c_used_elsewhere = True
                                break
                        
                        if c_used_elsewhere:
                            continue # Recoloring (w,x) won't free c at w
                        
                        # Check if c is used elsewhere at x? No, x is not a common neighbor of u,v necessarily?
                        # Wait, w is a common neighbor. The constraint is on w.
                        # So we just need to free c at w.
                        
                        # Perform the recolor
                        st.recolor(w, x, d)
                        
                        # Now try to add (u, v) with c
                        if st.valid_add(u, v, c):
                            st.add(u, v, c)
                            improved = True
                            found_recolor = True
                            break
                    
                    if found_recolor:
                        break
                
                if found_recolor:
                    break
    
    return st

def construct():
    start = time.time()
    deadline = start + DEADLINE
    best = None
    best_count = -1
    
    # Run multiple restarts
    # Each restart: Greedy + Local Search
    # We need to be fast. Greedy is O(N^2 * K * N) ~ 20^4 = 160k ops. Very fast.
    # Local search is the bottleneck.
    
    for seed in range(10000):
        if time.time() > deadline - 5.0:
            break
        
        rng = random.Random(seed)
        st = greedy_build(rng)
        
        # Spend some time on local search
        # Estimate time per local search step.
        # If we have 100s left, and we do 100 restarts, we have 1s per restart.
        # Let's give 0.5s to local search per restart.
        ls_deadline = time.time() + 0.5
        st = local_search(st, rng, ls_deadline)
        
        cnt = st.count()
        if cnt > best_count:
            best_count = cnt
            best = st
            
        # If we hit a high density, we can stop early or tighten search
        if best_count >= 180: # Heuristic upper bound guess
             break

    if best is None:
        # Fallback: empty graph or star
        return []
        
    # Final polish: try to add more edges
    final_deadline = deadline - 1.0
    best = local_search(best, random.Random(999), final_deadline)
    
    edges = [[u, v, c] for (u, v), c in best.edge.items()]
    return edges
