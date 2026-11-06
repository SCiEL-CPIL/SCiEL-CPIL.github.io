import time
import random

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin before 120s

    # Try to find a solution for n=26, then 25, etc.
    # The seed was n=11.
    # We know n=26 is possible.
    
    def count_violations(n, colors):
        """
        colors is a list of length n*(n-1)/2.
        Returns the number of C4s (each C4 counted twice in the sum of binom(k,2)).
        """
        # Map (i, j) to index
        def idx(i, j):
            if i > j: i, j = j, i
            return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
        
        # Build adjacency sets for each color
        neighbors = [set() for _ in range(n)] # neighbors[c][v] is set of neighbors in color c
        # Actually, we need neighbors per color per vertex.
        # neighbors[c][v]
        neighbors = [[set() for _ in range(n)] for _ in range(5)]
        
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx(i, j)]
                neighbors[c][i].add(j)
                neighbors[c][j].add(i)
                
        total_violations = 0
        for c in range(5):
            for u in range(n):
                for v in range(u + 1, n):
                    common = len(neighbors[c][u] & neighbors[c][v])
                    if common >= 2:
                        total_violations += common * (common - 1) // 2
        return total_violations

    def get_delta_v(n, colors, u, v, c_old, c_new):
        """
        Calculate change in violations if edge (u,v) changes from c_old to c_new.
        """
        def idx(i, j):
            if i > j: i, j = j, i
            return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
            
        # We need to know the neighbors.
        # For efficiency in a loop, we should maintain the neighbor sets.
        # But this function is called for evaluation.
        # Let's assume we have a state object.
        pass

    # Better approach: Maintain state explicitly.
    class State:
        def __init__(self, n):
            self.n = n
            self.M = n * (n - 1) // 2
            self.colors = [0] * self.M
            self.neighbors = [[set() for _ in range(n)] for _ in range(5)]
            self.violations = 0
            
        def idx(self, i, j):
            if i > j: i, j = j, i
            return i * (n - 1) - i * (i + 1) // 2 + (j - i - 1)
            
        def init_random(self):
            self.colors = [random.randint(0, 4) for _ in range(self.M)]
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    self.add_edge(i, j, self.colors[self.idx(i, j)])
            self.recalc_violations()
            
        def add_edge(self, i, j, c):
            self.neighbors[c][i].add(j)
            self.neighbors[c][j].add(i)
            
        def remove_edge(self, i, j, c):
            self.neighbors[c][i].discard(j)
            self.neighbors[c][j].discard(i)
            
        def recalc_violations(self):
            self.violations = 0
            for c in range(5):
                for u in range(self.n):
                    for v in range(u + 1, self.n):
                        common = len(self.neighbors[c][u] & self.neighbors[c][v])
                        if common >= 2:
                            self.violations += common * (common - 1) // 2
                            
        def get_delta(self, u, v, c_old, c_new):
            """
            Returns delta_violations for moving (u,v) from c_old to c_new.
            """
            if c_old == c_new:
                return 0
                
            delta = 0
            
            # 1. Remove from c_old
            # Affected pairs: (u, w) for w in N_c_old(v) \ {u}
            #               (v, w) for w in N_c_old(u) \ {v}
            # Note: The pair (u,v) itself is not in the violation sum for c_old 
            #       because violations are counted for pairs of vertices (a,b) 
            #       based on common neighbors. 
            #       Wait, my violation count is sum_{a<b} binom(|N(a) \cap N(b)|, 2).
            #       The edge (u,v) itself is not a pair of vertices in this sum.
            #       The sum is over all pairs of vertices in the graph.
            #       So (u,v) is a pair in the sum.
            #       k(u,v) = |N(u) \cap N(v)|.
            #       When we remove (u,v), N(u) and N(v) change.
            #       Does k(u,v) change?
            #       N(u) loses v. N(v) loses u.
            #       Intersection N(u) \cap N(v) loses any x such that x was in both.
            #       v is in N(u) but not in N(v) (no self loops).
            #       u is in N(v) but not in N(u).
            #       So v is not in N(u) \cap N(v).
            #       So k(u,v) does NOT change when removing (u,v).
            
            # So only pairs involving u or v, other than (u,v), change.
            # Pairs (u, w) where w != v.
            # k(u, w) = |N(u) \cap N(w)|.
            # N(u) changes: loses v.
            # So if v was in N(w), then v was in the intersection.
            # So k(u, w) decreases by 1 if v in N_c_old(w).
            # Similarly for (v, w) where w != u.
            # k(v, w) decreases by 1 if u in N_c_old(w).
            
            # Let's calculate the change in binom(k, 2) for these pairs.
            
            # For each w in N_c_old(v) (which are neighbors of v in c_old):
            # w != u (since (u,v) is an edge, u is in N(v), but we iterate N(v)).
            # Wait, N_c_old(v) contains u.
            # For w in N_c_old(v):
            #   if w == u: pair (u,v) is not affected in k? 
            #   We established k(u,v) doesn't change.
            #   So we skip w=u.
            
            # For w in N_c_old(v) \ {u}:
            #   Pair (u, w) is affected.
            #   k_old = |N_c_old(u) \cap N_c_old(w)|.
            #   Since v in N_c_old(u) and v in N_c_old(w) (because w in N_c_old(v) implies v in N_c_old(w) by symmetry),
            #   v was in the intersection.
            #   So k_new = k_old - 1.
            #   Delta for this pair = binom(k_new, 2) - binom(k_old, 2)
            #                      = (k_new*(k_new-1) - k_old*(k_old-1)) / 2
            #                      = ((k_old-1)(k_old-2) - k_old(k_old-1)) / 2
            #                      = (k_old-1) * (k_old-2 - k_old) / 2
            #                      = (k_old-1) * (-2) / 2
            #                      = -(k_old - 1)
            
            # Similarly for pairs (v, w) where w in N_c_old(u) \ {v}.
            # Delta = -(k_old(v,w) - 1)
            
            # 2. Add to c_new
            # N_c_new(u) gains v. N_c_new(v) gains u.
            # For w in N_c_new(v) \ {u} (before adding? No, after adding u is in N(v)).
            # We should evaluate based on the state BEFORE the move for c_new?
            # No, we are moving TO c_new.
            # Current state has (u,v) NOT in c_new.
            # So N_c_new(u) does NOT contain v.
            # When we add (u,v) to c_new:
            # N_c_new(u) gains v.
            # For any w in N_c_new(v) (current state, so w != u):
            #   Pair (u, w) is affected.
            #   k_old = |N_c_new(u) \cap N_c_new(w)|.
            #   Since v in N_c_new(v) and v will be in N_c_new(u),
            #   v will be in the new intersection.
            #   Was v in the old intersection?
            #   v was NOT in N_c_new(u) (old state).
            #   So v was NOT in the old intersection.
            #   So k_new = k_old + 1.
            #   Delta = binom(k_old+1, 2) - binom(k_old, 2)
            #         = ((k_old+1)k_old - k_old(k_old-1)) / 2
            #         = k_old * ( (k_old+1) - (k_old-1) ) / 2
            #         = k_old * 2 / 2
            #         = k_old
            
            # So for each w in N_c_new(v) \ {u}:
            #   Delta += k_c_new(u, w)
            
            # And for each w in N_c_new(u) \ {v}:
            #   Delta += k_c_new(v, w)
            
            # Implementation:
            
            # Remove from c_old
            # Iterate w in neighbors[c_old][v]
            for w in self.neighbors[c_old][v]:
                if w == u: continue
                # k_old(u, w)
                # We need to compute this.
                # |neighbors[c_old][u] & neighbors[c_old][w]|
                # This is O(n) per w. Total O(n^2) per move.
                # With n=26, this is 676 ops.
                # Million steps = 676M ops. Might be too slow in Python.
                # Optimization: Precompute intersection sizes?
                # Or just accept it and limit steps.
                
                k_old = len(self.neighbors[c_old][u] & self.neighbors[c_old][w])
                delta -= (k_old - 1)
                
            for w in self.neighbors[c_old][u]:
                if w == v: continue
                k_old = len(self.neighbors[c_old][v] & self.neighbors[c_old][w])
                delta -= (k_old - 1)
                
            # Add to c_new
            for w in self.neighbors[c_new][v]:
                if w == u: continue
                k_old = len(self.neighbors[c_new][u] & self.neighbors[c_new][w])
                delta += k_old
                
            for w in self.neighbors[c_new][u]:
                if w == v: continue
                k_old = len(self.neighbors[c_new][v] & self.neighbors[c_new][w])
                delta += k_old
                
            return delta

    def solve(n):
        if time.time() > deadline - 5:
            return None
            
        best_state = None
        best_violations = float('inf')
        
        # Multiple restarts
        for restart in range(20):
            if time.time() > deadline - 5:
                break
                
            state = State(n)
            state.init_random()
            
            # Local Search
            stuck = 0
            for step in range(50000):
                if time.time() > deadline - 5:
                    break
                    
                if state.violations == 0:
                    return {'n': n, 'colors': state.colors}
                    
                # Pick random edge
                idx = random.randint(0, state.M - 1)
                # Decode idx to u, v
                # idx = i*(n-1) - i*(i+1)//2 + (j-i-1)
                # This is hard to invert quickly.
                # Precompute edge list.
                
                # Let's just pick random u, v
                u = random.randint(0, n - 1)
                v = random.randint(0, n - 1)
                if u == v: continue
                if u > v: u, v = v, u
                e_idx = state.idx(u, v)
                c_old = state.colors[e_idx]
                
                best_delta = 0
                best_c_new = c_old
                
                for c_new in range(5):
                    if c_new == c_old: continue
                    d = state.get_delta(u, v, c_old, c_new)
                    if d < best_delta:
                        best_delta = d
                        best_c_new = c_new
                        
                if best_delta < 0:
                    # Apply move
                    state.remove_edge(u, v, c_old)
                    state.add_edge(u, v, best_c_new)
                    state.colors[e_idx] = best_c_new
                    state.violations += best_delta
                    stuck = 0
                else:
                    stuck += 1
                    if stuck > 1000:
                        break
                        
            if state.violations < best_violations:
                best_violations = state.violations
                best_state = state
                
        if best_violations == 0:
            return {'n': n, 'colors': best_state.colors}
        return None

    # Try n=26
    res = solve(26)
    if res:
        return res
        
    # Try n=25
    if time.time() < deadline - 10:
        res = solve(25)
        if res:
            return res
            
    # Fallback to seed n=11
    n = 11
    return {'n': n, 'colors': [min(j - i, n - j + i) - 1 for i in range(n) for j in range(i + 1, n)]}
