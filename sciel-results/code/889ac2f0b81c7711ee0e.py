"""Program for R_5(C4) maximization: greedy randomized construction with
incremental C4 counting, followed by fast local search."""
import time
import random
import numpy as np

def construct():
    """
    Returns {'n': n, 'colors': [...]} where colors is a list of length n*(n-1)//2
    giving one colour in {0,...,4} for each pair (i,j), i<j, in lexicographic order.
    No colour class may contain a C4 (no two vertices with >=2 common neighbours in one colour).
    """
    t0 = time.time()
    deadline = t0 + 115.0  # safety margin before 120s kill

    # ---------- Edge indexing helpers ----------

    def edge_idx(i, j, n):
        """Lexicographic index of edge (i,j) with i<j in a list of length n*(n-1)//2."""
        return i * n - i * (i + 1) // 2 - 1 + (j - i - 1)

    def idx_to_edge(idx, n):
        """Reverse: given linear index idx (0-based), return (i,j) with i<j."""
        row_start = 0
        for i in range(n):
            row_len = n - 1 - i
            if idx < row_start + row_len:
                j = i + 1 + (idx - row_start)
                return (i, j)
            row_start += row_len
        return (0, 1)

    # ---------- Incremental C4 counter ----------
    # common[c][u][v] = number of common neighbors of u and v in color c (u != v)
    # total_c4 = sum over all c, all unordered pairs {u,v} of max(0, common[c][u][v] - 1)
    #           = sum over all c, all unordered pairs {u,v} of (common[c][u][v] - 1) when common >= 2

    class C4Counter:
        def __init__(self, n):
            self.n = n
            self.m = n * (n - 1) // 2
            # colors[eidx] for eidx in 0..m-1
            self.colors = [-1] * self.m
            # neighbors[c][u] = list of neighbors of u in color c
            self.neighbors = [[[] for _ in range(n)] for _ in range(5)]
            # common[c][u][v] for u < v: number of common neighbors in color c
            # Use a flat array for speed: common[c][u*n + v]
            self.common = [np.zeros(n * n, dtype=np.int32) for _ in range(5)]
            self.total = 0  # total excess C4 count

        def _common_val(self, c, u, v):
            if u > v:
                u, v = v, u
            return self.common[c][u * self.n + v]

        def _set_common(self, c, u, v, val):
            if u > v:
                u, v = v, u
            self.common[c][u * self.n + v] = val

        def _add_common(self, c, u, v, delta):
            if u > v:
                u, v = v, u
            old = self.common[c][u * self.n + v]
            new = old + delta
            self.common[c][u * self.n + v] = new
            # Update total
            old_excess = max(0, old - 1)
            new_excess = max(0, new - 1)
            self.total += new_excess - old_excess

        def assign(self, eidx, c):
            """Assign color c to edge eidx (assuming it was unassigned or we handle removal first)."""
            n = self.n
            i, j = idx_to_edge(eidx, n)
            old_c = self.colors[eidx]
            if old_c == c:
                return
            if old_c != -1:
                self._remove_edge(i, j, old_c)
            self._add_edge(i, j, c)
            self.colors[eidx] = c

        def _add_edge(self, u, v, c):
            """Add edge (u,v) in color c, updating common counts."""
            n = self.n
            # For each w != u, v: common[c][u][w] increases by 1 (v is now a common neighbor)
            # and common[c][v][w] increases by 1 (u is now a common neighbor)
            # Also common[c][u][v] itself: the edge (u,v) doesn't directly affect common[u][v]
            # but we need to be careful.
            # Actually: adding edge (u,v) in color c means:
            #   - For each w in neighbors[c][u] (w != v): common[c][w][v] += 1 (u is now common neighbor of w and v)
            #   - For each w in neighbors[c][v] (w != u): common[c][u][w] += 1 (v is now common neighbor of u and w)
            #   - common[c][u][v] += 0? No. common[u][v] counts common neighbors. Adding edge (u,v) doesn't add a common neighbor to u and v.
            # Wait, I need to reconsider. common[c][u][v] = |N_c(u) ∩ N_c(v)|.
            # When we add edge (u,v) in color c:
            #   N_c(u) gains v, N_c(v) gains u.
            #   For any w not in {u,v}: 
            #     if w in N_c(u) before: v is now in N_c(w)? No, v is in N_c(u) but that doesn't mean u is in N_c(v) for w's perspective.
            #   Let me think again. common[c][u][v] = |N_c(u) ∩ N_c(v)|.
            #   Before adding (u,v): N_c(u) and N_c(v) don't include each other.
            #   After: N_c(u) has v added, N_c(v) has u added.
            #   N_c(u) ∩ N_c(v) after = (N_c(u) ∪ {v}) ∩ (N_c(v) ∪ {u})
            #   = (N_c(u) ∩ N_c(v)) ∪ (N_c(u) ∩ {u}) ∪ ({v} ∩ N_c(v)) ∪ ({v} ∩ {u})
            #   = (N_c(u) ∩ N_c(v)) ∪ (∅) ∪ ({v} if v in N_c(v) else ∅) ∪ (∅)
            #   Hmm, v is in N_c(v)? No, no self-loops. So {v} ∩ N_c(v) = ∅.
            #   Similarly {u} ∩ N_c(u) = ∅.
            #   So common[c][u][v] doesn't change when adding edge (u,v)!
            #   
            #   But for pairs involving u and some w (w not in {u,v}):
            #   common[c][u][w] = |N_c(u) ∩ N_c(w)|.
            #   N_c(u) gains v. So if v in N_c(w), then common[c][u][w] increases by 1.
            #   Similarly, common[c][v][w] = |N_c(v) ∩ N_c(w)|.
            #   N_c(v) gains u. So if u in N_c(w), then common[c][v][w] increases by 1.
            #
            #   So: for each w in N_c(u) (w != v): common[c][v][w] += 1
            #   And: for each w in N_c(v) (w != u): common[c][u][w] += 1
            
            # First, update common counts
            for w in self.neighbors[c][u]:
                if w != v:
                    self._add_common(c, v, w, 1)
            for w in self.neighbors[c][v]:
                if w != u:
                    self._add_common(c, u, w, 1)
            
            # Now add to neighbor lists
            self.neighbors[c][u].append(v)
            self.neighbors[c][v].append(u)

        def _remove_edge(self, u, v, c):
            """Remove edge (u,v) from color c."""
            n = self.n
            # Reverse of _add_edge
            for w in self.neighbors[c][u]:
                if w != v:
                    self._add_common(c, v, w, -1)
            for w in self.neighbors[c][v]:
                if w != u:
                    self._add_common(c, u, w, -1)
            
            # Remove from neighbor lists
            self.neighbors[c][u].remove(v)
            self.neighbors[c][v].remove(u)

        def get_total(self):
            return self.total

    # ---------- Greedy randomized construction ----------

    def greedy_construct(n, seed=42, time_limit=10.0):
        """Greedy: process edges in random order, assign color minimizing new C4s."""
        random.seed(seed)
        m = n * (n - 1) // 2
        edges = list(range(m))
        random.shuffle(edges)
        
        counter = C4Counter(n)
        t_start = time.time()
        
        for eidx in edges:
            if time.time() > t_start + time_limit:
                break
            i, j = idx_to_edge(eidx, n)
            
            # For each color c, compute how many new C4s would be created
            best_c = -1
            best_delta = float('inf')
            candidates = []
            
            for c in range(5):
                # Compute delta: sum over w in N_c(u) (w != v) of 1 + sum over w in N_c(v) (w != u) of 1
                # But we also need to account for the change in excess, not just common count.
                # Simpler: just count new common-neighbor additions that push from 1 to 2.
                # For approximation, just count the number of w where common goes from 1 to 2.
                delta = 0
                for w in counter.neighbors[c][i]:
                    if w != j:
                        old = counter._common_val(c, j, w)
                        if old == 1:
                            delta += 1
                for w in counter.neighbors[c][j]:
                    if w != i:
                        old = counter._common_val(c, i, w)
                        if old == 1:
                            delta += 1
                
                candidates.append((delta, c))
            
            # Pick minimum delta, break ties randomly
            candidates.sort()
            min_delta = candidates[0][0]
            min_candidates = [c for d, c in candidates if d == min_delta]
            best_c = random.choice(min_candidates)
            
            counter.assign(eidx, best_c)
        
        return counter.colors, counter.get_total()

    # ---------- Fast local search with incremental counter ----------

    def local_search_fast(n, colors_init, time_limit=30.0, seed=12345):
        """Local search using incremental C4 counter."""
        random.seed(seed)
        m = n * (n - 1) // 2
        counter = C4Counter(n)
        
        # Initialize counter from colors_init
        for eidx in range(m):
            if colors_init[eidx] != -1:
                counter.assign(eidx, colors_init[eidx])
        
        current_total = counter.get_total()
        if current_total == 0:
            return counter.colors, True
        
        t_start = time.time()
        edges = list(range(m))
        
        while time.time() < t_start + time_limit:
            if current_total == 0:
                return counter.colors, True
            
            # Pick random edge
            eidx = random.randint(0, m - 1)
            old_c = counter.colors[eidx]
            if old_c == -1:
                continue
            
            # Try each other color
            i, j = idx_to_edge(eidx, n)
            best_new_c = old_c
            best_delta = 0  # 0 means keep old
            
            for new_c in range(5):
                if new_c == old_c:
                    continue
                # Compute delta if we change from old_c to new_c
                # Remove from old_c: for w in N_old(i) (w!=j): common_old[j][w] -= 1
                #                 for w in N_old(j) (w!=i): common_old[i][w] -= 1
                # Add to new_c:    for w in N_new(i) (w!=j): common_new[j][w] += 1
                #                 for w in N_new(j) (w!=i): common_new[i][w] += 1
                delta = 0
                for w in counter.neighbors[old_c][i]:
                    if w != j:
                        old_val = counter._common_val(old_c, j, w)
                        old_excess = max(0, old_val - 1)
                        new_excess = max(0, old_val - 2)
                        delta += new_excess - old_excess
                for w in counter.neighbors[old_c][j]:
                    if w != i:
                        old_val = counter._common_val(old_c, i, w)
                        old_excess = max(0, old_val - 1)
                        new_excess = max(0, old_val - 2)
                        delta += new_excess - old_excess
                for w in counter.neighbors[new_c][i]:
                    if w != j:
                        old_val = counter._common_val(new_c, j, w)
                        old_excess = max(0, old_val - 1)
                        new_excess = max(0, old_val)
                        delta += new_excess - old_excess
                for w in counter.neighbors[new_c][j]:
                    if w != i:
                        old_val = counter._common_val(new_c, i, w)
                        old_excess = max(0, old_val - 1)
                        new_excess = max(0, old_val)
                        delta += new_excess - old_excess
                
                if delta < best_delta:
                    best_delta = delta
                    best_new_c = new_c
            
            if best_new_c != old_c:
                counter.assign(eidx, best_new_c)
                current_total = counter.get_total()
        
        return counter.colors, (current_total == 0)

    # ---------- Main logic ----------

    # Try n=26 with greedy + local search
    for target_n in [26, 25, 24]:
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        if remaining < 5:
            break
        
        # Greedy construction
        greedy_time = min(8.0, remaining * 0.3)
        colors_greedy, total_greedy = greedy_construct(target_n, seed=42, time_limit=greedy_time)
        
        if total_greedy == 0:
            return {'n': target_n, 'colors': colors_greedy}
        
        # Local search from greedy result
        ls_time = min(remaining * 0.5, 40.0)
        colors_ls, valid_ls = local_search_fast(target_n, colors_greedy, time_limit=ls_time, seed=12345)
        
        if valid_ls:
            return {'n': target_n, 'colors': colors_ls}
        
        # Try a few more restarts with different seeds
        for restart_seed in [100, 200, 300]:
            if time.time() > deadline:
                break
            remaining = deadline - time.time()
            if remaining < 3:
                break
            
            greedy_time = min(5.0, remaining * 0.4)
            colors_g, total_g = greedy_construct(target_n, seed=restart_seed, time_limit=greedy_time)
            
            if total_g == 0:
                return {'n': target_n, 'colors': colors_g}
            
            ls_time = min(remaining * 0.5, 20.0)
            colors_l, valid_l = local_search_fast(target_n, colors_g, time_limit=ls_time, seed=restart_seed + 1000)
            
            if valid_l:
                return {'n': target_n, 'colors': colors_l}
    
    # Fallback: deterministic small constructions
    # n=11 cyclic: color(i,j) = (j-i) % 11 mapped to 5 colors
    # Use a known valid small construction
    for fallback_n in [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]:
        if time.time() > deadline:
            break
        m = fallback_n * (fallback_n - 1) // 2
        colors = [0] * m
        for idx in range(m):
            i, j = idx_to_edge(idx, fallback_n)
            colors[idx] = (j - i) % 5
        # Verify
        counter = C4Counter(fallback_n)
        for eidx in range(m):
            counter.assign(eidx, colors[eidx])
        if counter.get_total() == 0:
            return {'n': fallback_n, 'colors': colors}
    
    # Absolute last resort: n=1
    return {'n': 1, 'colors': []}
