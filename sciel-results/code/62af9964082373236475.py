import time
import random
import sys

def construct():
    """
    Finds a 5-coloring of K_n without monochromatic C4.
    Maximizes n.
    """
    deadline = time.time() + 110.0  # Safety margin for 120s limit
    
    # Known seed solution for n=11 (cyclic group Z11, colors based on difference)
    def get_seed_solution():
        n = 11
        colors = []
        for i in range(n):
            for j in range(i + 1, n):
                d = min(j - i, n - (j - i))
                colors.append(d - 1)
        return n, colors

    best_n, best_colors = get_seed_solution()
    
    # Helper to get edge index in lexicographic order
    def get_edge_index(i, j, n):
        if i > j:
            i, j = j, i
        return i * (i - 1) // 2 + (j - i - 1)

    # Incremental C4 checker
    # neighbors[c][u] is a set of v such that edge (u,v) is color c
    class C4Checker:
        def __init__(self, n):
            self.n = n
            self.neighbors = [[set() for _ in range(5)] for _ in range(n)]
            self.violations = 0 # Count of pairs (u,v) with >=2 common neighbors in any color
            self.violation_pairs = [] # List of (u, v, c) that are currently violating
            
        def add_edge(self, u, v, c):
            self.neighbors[u][c].add(v)
            self.neighbors[v][c].add(u)
            # Check if this addition creates a new violation
            # A violation is a pair (u,v) with >=2 common neighbors in color c.
            # Adding edge (u,v) with color c affects:
            # 1. Pairs involving u and its neighbors in c? No, common neighbors of u and x.
            # Actually, it's easier to just check all pairs involving u or v?
            # For speed, we can maintain a count of common neighbors for each pair?
            # That is O(n^2) memory. n=29, 29*29/2 = 400 pairs. 5 colors. 2000 counters.
            # Let's maintain common_count[c][u][v] = number of common neighbors.
            # But updating this is complex.
            # Given n is small (<=29), we can afford a slightly slower check on update.
            # When adding (u,v,c):
            # For any x in neighbors[u][c] (before adding v? no, after adding v, v is in neighbors[u][c])
            # The new edge (u,v) means v is a neighbor of u.
            # Does this create a C4?
            # A C4 involves u, v, x, y.
            # If we just added (u,v), the cycle must use this edge.
            # Cycle: u-v-x-y-u.
            # So x must be a neighbor of v in color c, and y must be a neighbor of u in color c.
            # And x,y must be connected? No, the condition is just that u and v share 2 neighbors.
            # Wait, the condition "no two vertices with two common neighbors" is equivalent to no C4.
            # If u and v have 2 common neighbors x,y, then u-x-v-y-u is a C4.
            # So, adding edge (u,v,c) increases the common neighbor count of pair (u,v) by 0?
            # No, it increases the common neighbor count of pairs (u, x) and (v, y) where x~v and y~u?
            
            # Let's stick to a simpler robust method for n<=29:
            # On each assignment, we don't update a global violation count incrementally in a complex way.
            # Instead, we maintain the neighbor sets.
            # We check for violations by scanning? No, that's O(n^2) per step.
            # 400 steps * 400 ops = 160,000 ops. Very fast.
            # But we need to know IF a violation exists to backtrack.
            # We can just check if ANY violation exists after each move.
            # To make it faster, we can check only pairs involving the new edge endpoints?
            # If we add edge (u,v,c), it can only create new violations for pairs that include u or v?
            # Let common(u, v) be the set of common neighbors.
            # Adding edge (u,v,c) means v is added to N_c(u) and u to N_c(v).
            # This changes common(u, w) for all w?
            # common(u, w) = N_c(u) ∩ N_c(w).
            # New N_c(u) = Old N_c(u) ∪ {v}.
            # New common(u, w) = (Old N_c(u) ∪ {v}) ∩ N_c(w) = Old common(u, w) ∪ ({v} ∩ N_c(w)).
            # So common(u, w) increases by 1 if v is in N_c(w).
            # Similarly, common(v, w) increases by 1 if u is in N_c(w).
            # So, adding edge (u,v,c) only affects the common-neighbor counts of pairs (u, w) and (v, w) for all w.
            # Specifically, it increases the count for (u, w) if w is a c-neighbor of v.
            # And for (v, w) if w is a c-neighbor of u.
            
            # So, we can maintain a 2D array common_count[c][u][v] for u < v?
            # Or just a dict/set of violating pairs.
            
            # Let's maintain a list of "bad" pairs.
            # A pair (u,v) is bad if |N_c(u) ∩ N_c(v)| >= 2.
            
            # To update efficiently:
            # When adding (u,v,c):
            # 1. v is added to N_c(u).
            #    This increases |N_c(u) ∩ N_c(w)| by 1 for all w in N_c(v).
            #    So for each w in N_c(v), the pair (u,w) becomes "more bad".
            #    If it was 1, it becomes 2 (new violation).
            #    If it was >=2, it was already a violation.
            # 2. u is added to N_c(v).
            #    This increases |N_c(v) ∩ N_c(w)| by 1 for all w in N_c(u).
            #    So for each w in N_c(u), the pair (v,w) becomes "more bad".
            
            # We need to track the count of common neighbors for each pair.
            # common_count[c][u][v] for u != v.
            
            # Let's implement this.
            
            # For now, in this simple class, I will just do a full check if needed, 
            # but for the SAT solver, I will use the incremental logic directly.
            pass

    # SAT Solver for 5-coloring K_n without C4
    # Variables: x[i][j] for i<j, value in {0,1,2,3,4}.
    # Constraints: For each c, for all u<v, |N_c(u) ∩ N_c(v)| <= 1.
    
    def solve_n(n, time_limit):
        """
        Solves for a valid coloring of K_n using DPLL with forward checking.
        Returns colors list or None.
        """
        total_edges = n * (n - 1) // 2
        # Assignment array: -1 for unassigned
        assignment = [-1] * total_edges
        
        # Neighbors sets
        neighbors = [[set() for _ in range(5)] for _ in range(n)]
        
        # Common neighbor counts: common_count[c][u][v]
        # We only care if it reaches 2.
        # Initialize to 0.
        # To save memory and time, we can use a dict or just a 3D list.
        # n=29, 5*29*29 is small.
        common_count = [[[0 for _ in range(n)] for _ in range(n)] for _ in range(5)]
        
        # List of violating pairs (u, v, c) where common_count[c][u][v] >= 2
        # We maintain a set of active violations to check if solution is valid.
        # Actually, we can just check if any count is >= 2.
        # To make backtracking fast, we keep a list of counts that are currently >= 2.
        # But we need to know which ones to decrement on backtrack.
        
        # Let's store the history of changes to common_count to allow undo.
        # Or, simpler: since n is small, we can just recompute? No, that's slow.
        # We will store the delta changes.
        
        # Variable ordering: Heuristic. 
        # Start with edges that are most constrained? 
        # Or just lexicographic? 
        # Let's try a random order or a specific structure.
        # For n=26, the Lazebnik-Woldar construction is based on Z_26? No, it's a specific algebraic construction.
        # We can try to use a "cyclic" initialization for n=11, but for larger n, random is safer.
        
        # To improve success, we can try multiple random restarts with different orderings.
        
        def get_idx(i, j):
            if i > j: i, j = j, i
            return i * (i - 1) // 2 + (j - i - 1)
            
        # Precompute edge endpoints
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                edges.append((i, j))
                
        # Randomize edge order for diversity
        random.shuffle(edges)
        
        # Map edge index to position in edges list for quick access?
        # We iterate through edges in the shuffled order.
        # We need to assign colors to edges in this order.
        
        # Stack for backtracking
        # State: (edge_index_in_edges, color_assigned, previous_common_counts_snapshot?)
        # Snapshotting common_counts is expensive.
        # Instead, we record the changes made to common_counts and neighbors, and undo them.
        
        # Let's define a function to apply an assignment and return a list of changes.
        def apply_assignment(u, v, c):
            changes = []
            # Update neighbors
            neighbors[u][c].add(v)
            neighbors[v][c].add(u)
            
            # Update common counts
            # For each w in neighbors[v][c] (before adding u? No, u is added to neighbors[v][c] AFTER)
            # Wait, the logic:
            # Adding (u,v,c):
            # 1. v added to N_c(u).
            #    For each w in N_c(v) (current, which does NOT include u yet), 
            #    common_count[c][u][w] += 1.
            # 2. u added to N_c(v).
            #    For each w in N_c(u) (current, which does NOT include v yet),
            #    common_count[c][v][w] += 1.
            
            # We must be careful with the order of iteration.
            # Let's snapshot the relevant sets first.
            N_v_c = list(neighbors[v][c])
            N_u_c = list(neighbors[u][c])
            
            # Apply 1: v added to N_c(u)
            for w in N_v_c:
                if w == u: continue # Should not be there
                # Update common_count[c][u][w]
                old = common_count[c][u][w]
                common_count[c][u][w] = old + 1
                # Symmetry: common_count[c][w][u]
                common_count[c][w][u] = old + 1
                changes.append(('cc', c, u, w, 1))
                
            # Apply 2: u added to N_c(v)
            for w in N_u_c:
                if w == v: continue
                old = common_count[c][v][w]
                common_count[c][v][w] = old + 1
                common_count[c][w][v] = old + 1
                changes.append(('cc', c, v, w, 1))
                
            # Update neighbor sets
            neighbors[u][c].add(v)
            neighbors[v][c].add(u)
            changes.append(('nb', u, c, v, 1))
            changes.append(('nb', v, c, u, 1))
            
            return changes

        def undo_assignment(changes):
            for change in reversed(changes):
                if change[0] == 'cc':
                    _, c, u, w, delta = change
                    common_count[c][u][w] -= delta
                    common_count[c][w][u] -= delta
                elif change[0] == 'nb':
                    _, u, c, v, delta = change
                    if delta == 1:
                        neighbors[u][c].discard(v)
        
        def has_violation():
            # Check if any common_count[c][u][v] >= 2
            # We can maintain a counter of violations?
            # Or just scan? n=29, 5*29*29 = 4205 checks. Fast.
            # But we can be smarter.
            # We know which pairs were incremented.
            # If we maintain a list of "potentially violating" pairs, we can check them.
            # For now, scan is fine for n<=29.
            for c in range(5):
                for u in range(n):
                    for v in range(u + 1, n):
                        if common_count[c][u][v] >= 2:
                            return True
            return False
            
        # Optimization: Maintain a list of active violations?
        # Let's just use the scan for simplicity first, as it's O(n^2) and n is small.
        
        # DPLL
        def search(depth):
            if time.time() > time_limit:
                return False
            
            if depth == len(edges):
                # All edges assigned
                if not has_violation():
                    return True
                return False
            
            u, v = edges[depth]
            
            # Try colors 0-4
            # Heuristic: Try colors that are less likely to cause violations?
            # Or random order?
            colors_to_try = [0, 1, 2, 3, 4]
            random.shuffle(colors_to_try)
            
            for c in colors_to_try:
                # Forward checking:
                # If we assign c to (u,v), does it immediately create a violation?
                # It creates a violation if:
                # 1. There is a w such that common_count[c][u][w] is currently 1.
                #    Because adding (u,v) increments common_count[c][u][w] for w in N_c(v).
                #    If common_count[c][u][w] was 1, it becomes 2. Violation.
                # 2. Similarly for v.
                
                # Check condition 1:
                valid = True
                for w in neighbors[v][c]:
                    if w == u: continue
                    if common_count[c][u][w] >= 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # Check condition 2:
                for w in neighbors[u][c]:
                    if w == v: continue
                    if common_count[c][v][w] >= 1:
                        valid = False
                        break
                if not valid:
                    continue
                
                # If valid, apply and recurse
                changes = apply_assignment(u, v, c)
                assignment[get_idx(u, v)] = c
                
                if search(depth + 1):
                    return True
                
                undo_assignment(changes)
                assignment[get_idx(u, v)] = -1
                
            return False
            
        # Start search
        # We can try multiple random shuffles of edges
        # But the shuffle is done once above.
        # To allow multiple restarts, we should re-shuffle.
        
        # Let's wrap this in a restart loop outside?
        # The function solve_n currently does one run.
        # We will call it multiple times from construct().
        
        if search(0):
            # Build colors list in lexicographic order
            colors = [0] * total_edges
            for i in range(n):
                for j in range(i + 1, n):
                    colors[get_idx(i, j)] = assignment[get_idx(i, j)]
            return colors
        return None

    # Try to find a solution for n from 29 down to 12
    # We know n=26 is possible.
    # We will allocate time per n.
    
    # Time allocation strategy:
    # n=29, 28, 27: 5s each (likely impossible or very hard)
    # n=26: 30s (known possible, use SAT to find it reliably)
    # n=25..12: 5s each (easier, but we want max n, so if 26 fails, try 25)
    
    # Actually, if SAT works, it should find 26 quickly if we have a good heuristic.
    # But SAT on 325 variables (n=26) with 5 values is huge.
    # The forward checking prunes a lot.
    
    # Let's try n=26 first with a generous time, then higher ones?
    # No, we want MAX n. So try 29, 28, 27, 26...
    
    # However, SAT might get stuck in deep backtracking.
    # We will use random restarts.
    
    # To improve the SAT solver, we can use a better variable ordering.
    # "Most Constrained Variable" (MCV) or "Least Constraining Value" (LCV).
    # Dynamic variable ordering is complex to implement correctly with backtracking.
    # Static random ordering with restarts is a good baseline.
    
    # Let's modify solve_n to accept a seed for randomness and a time limit.
    
    def solve_n_with_restarts(n, total_time):
        start_time = time.time()
        # Try a few restarts
        # Each restart gets a portion of the time
        # Or just run until time runs out.
        
        # We will do a single long run with random restarts inside?
        # No, the current solve_n does one run.
        # We can call it multiple times.
        
        # Let's estimate: if n=26 is hard, we need many restarts.
        # Let's allow up to 10 restarts, each with (total_time / 10) limit.
        
        num_restarts = 10
        time_per_restart = total_time / num_restarts
        
        for i in range(num_restarts):
            if time.time() > start_time + total_time:
                break
            # Re-seed random for different edge order
            random.seed(i * 12345 + n)
            res = solve_n(n, start_time + time_per_restart * (i + 1))
            if res:
                return res
        return None

    # Execute search
    # Prioritize n=26 as it's the known max.
    # If we find 26, we can try 27, 28, 29 with remaining time?
    # But if 26 takes 30s, we have 80s left.
    # If we try 29 first, it might take 10s and fail.
    
    # Strategy:
    # 1. Try n=29, 28, 27 with 5s each.
    # 2. If any found, return it.
    # 3. Try n=26 with 40s.
    # 4. If found, return it.
    # 5. Try n=25 with 10s.
    # 6. ...
    # 7. Return seed (11).
    
    # But wait, the seed is 11. If we find 26, score is 26/11 = 2.36.
    # If we find 27, score is 2.45.
    
    # Let's try high to low.
    
    for n in range(29, 11, -1):
        if time.time() > deadline - 5:
            break
            
        if n >= 27:
            t_alloc = 5.0
        elif n == 26:
            t_alloc = 40.0
        else:
            t_alloc = 10.0
            
        # Ensure we don't exceed deadline
        t_alloc = min(t_alloc, deadline - time.time() - 2.0)
        if t_alloc <= 0:
            break
            
        # Run solver
        res = solve_n_with_restarts(n, t_alloc)
        
        if res:
            # Verify solution just in case
            # (The SAT solver guarantees it, but a quick check is safe)
            if is_valid(n, res):
                best_n = n
                best_colors = res
                # Since we go high to low, this is the max found so far.
                # We can return immediately if we want to save time?
                # Yes, because we are searching downwards.
                return {'n': best_n, 'colors': best_colors}
                
    # If all fail, return seed
    return {'n': best_n, 'colors': best_colors}

    # Helper for verification
    def is_valid(n, colors):
        neighbors = [[set() for _ in range(5)] for _ in range(n)]
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                c = colors[idx]
                if c < 0 or c > 4:
                    return False
                neighbors[i][c].add(j)
                neighbors[j][c].add(i)
                idx += 1
        
        for c in range(5):
            for u in range(n):
                nu = neighbors[u][c]
                if len(nu) < 2:
                    continue
                for v in range(u + 1, n):
                    nv = neighbors[v][c]
                    common = nu & nv
                    if len(common) >= 2:
                        return False
        return True

if __name__ == "__main__":
    # Quick test
    res = construct()
    print(f"Found n={res['n']}")
