import time
import random
import itertools

N = 16
NUM_EDGES = N * (N - 1) // 2
MAX_COLORS = 120

def construct():
    start_time = time.time()
    deadline = start_time + 55.0  # Safety margin before 60s limit
    
    # Precompute edge indices
    edge_idx = [[-1]*N for _ in range(N)]
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            edge_idx[i][j] = idx
            edge_idx[j][i] = idx
            idx += 1

    # Precompute cycles and their edge indices
    # cycles: list of tuples of 4 edge indices
    cycles = []
    # Map edge index to list of cycle indices containing it
    edge_to_cycles = [[] for _ in range(NUM_EDGES)]
    
    for quad in itertools.combinations(range(N), 4):
        a, b, c, d = quad
        # Cycle 1: a-b-c-d-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][c]
        e3 = edge_idx[c][d]
        e4 = edge_idx[d][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)
        
        # Cycle 2: a-b-d-c-a
        e1 = edge_idx[a][b]
        e2 = edge_idx[b][d]
        e3 = edge_idx[d][c]
        e4 = edge_idx[c][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)
        
        # Cycle 3: a-c-b-d-a
        e1 = edge_idx[a][c]
        e2 = edge_idx[c][b]
        e3 = edge_idx[b][d]
        e4 = edge_idx[d][a]
        cid = len(cycles)
        cycles.append((e1, e2, e3, e4))
        edge_to_cycles[e1].append(cid)
        edge_to_cycles[e2].append(cid)
        edge_to_cycles[e3].append(cid)
        edge_to_cycles[e4].append(cid)

    def is_bad_cycle(colors, c_idx):
        e1, e2, e3, e4 = c_idx
        c1 = colors[e1]
        c2 = colors[e2]
        c3 = colors[e3]
        c4 = colors[e4]
        return c1 != c2 and c1 != c3 and c1 != c4 and c2 != c3 and c2 != c4 and c3 != c4

    # --- Phase 1: Backtracking with MRV and 'New Color First' ---
    
    # We will maintain a list of colors for each edge. -1 means unassigned.
    # We need to track which colors are currently used to prioritize new ones.
    # For speed in backtracking, we'll do a randomized DFS.
    
    def backtrack_search(max_depth=50, time_limit=20.0):
        """
        Attempts to find a valid coloring with high diversity using randomized backtracking.
        Returns a valid coloring list or None.
        """
        colors = [-1] * NUM_EDGES
        used_colors = set()
        
        # Precompute constraints for faster checking? 
        # The cycle check is O(1) per cycle. 
        # When we assign color C to edge E, we only need to check cycles containing E.
        
        # To implement MRV, we need to know which edges are unassigned and how many valid colors they have.
        # Calculating "valid colors" for an edge involves checking all cycles it belongs to.
        # This is expensive to do for all edges at every step.
        
        # Alternative: Randomized Backtracking with Conflict-Driven Restarts.
        # We pick an unassigned edge randomly (or with some heuristic), try colors (new first), 
        # and if we hit a dead end, we backtrack.
        
        # A simpler, effective heuristic for this specific problem:
        # 1. Order edges randomly.
        # 2. For each edge, try unused colors first, then used colors.
        # 3. If a choice leads to a contradiction (bad cycle formed), backtrack.
        
        # However, pure backtracking on 120 edges might be slow if not guided.
        # Let's use a "Lookahead" approach: 
        # Pick the unassigned edge that is in the most "constrained" cycles?
        # Or just random order with random color order (biased to new).
        
        # Let's try a depth-first search with random restarts.
        
        edges_order = list(range(NUM_EDGES))
        random.shuffle(edges_order)
        
        # We'll use a stack for DFS
        # State: (edge_index_in_order, current_colors, used_colors)
        # This is memory heavy. Let's do recursive with backtracking.
        
        # To speed up, we can pre-compute for each edge, the list of cycles it's in.
        # edge_to_cycles is already there.
        
        # Let's limit the search depth or time.
        
        def dfs(idx):
            if time.time() > time_limit:
                return False
            
            if idx == len(edges_order):
                return True
            
            e = edges_order[idx]
            
            # Generate candidate colors
            # 1. Unused colors (sorted randomly)
            # 2. Used colors (sorted randomly)
            
            # To avoid building large lists, we can pick a random unused color first.
            # But we need to try them in a specific order to be systematic?
            # Randomized backtracking just picks a random valid color, or tries all.
            
            # Let's try: 
            # 1. Pick a random unused color if any exist.
            # 2. If that fails, try other unused colors.
            # 3. If all unused fail, try used colors.
            
            # Actually, for MRV, we should pick the edge with fewest options.
            # But calculating options is expensive.
            # Let's stick to random edge order, but smart color order.
            
            # Get list of unused colors
            # This is O(MAX_COLORS) per step. 120 * 120 = 14400 ops per level. 
            # Depth 120. 1.7M ops per path. With backtracking, this could be slow.
            
            # Optimization: Maintain a list of unused colors.
            # When we assign a new color, remove it from the list.
            # When we backtrack, add it back.
            
            # Let's implement a specific backtracking routine.
            
            # We'll use a global list for colors and a set for used.
            # We pass the index in the edges_order.
            
            # To make it faster, let's pre-shuffle the colors 0..119.
            # But "New First" means we want to try colors not in `used_colors`.
            
            # Let's try a simpler approach: 
            # Randomly assign colors to edges, but check validity immediately.
            # If invalid, change the last assigned edge to a different color.
            
            # This is essentially backtracking.
            
            # Let's implement a recursive function that takes the current state.
            # To avoid deep recursion limits and overhead, we'll use a stack.
            
            # Actually, for N=16, the constraint graph is dense.
            # Let's try a "Greedy with Random Restarts" first, which is simpler.
            # 1. Shuffle edges.
            # 2. For each edge, pick a random color from 0..119.
            # 3. If it creates a bad cycle, reject and pick another.
            # 4. If no color works, fail.
            # 5. If successful, we have a valid coloring.
            # 6. To maximize diversity, bias towards unused colors.
            
            pass # Placeholder for now, we will implement the actual logic below

        # --- Implementation of Backtracking with MRV-like Heuristic ---
        
        # We will maintain:
        # colors: list of current assignments
        # used_colors: set of colors currently in use
        # unassigned_edges: list of edges not yet assigned
        
        # We pick an edge from unassigned_edges.
        # Heuristic: Pick the edge that appears in the most cycles involving other unassigned edges?
        # Too complex. Let's just pick randomly from unassigned, but try colors in order:
        # 1. Unused colors (random order)
        # 2. Used colors (random order)
        
        # If we hit a dead end (no valid color for an edge), we backtrack.
        
        # To make this efficient, we need to quickly check if a color is valid for an edge.
        # A color C is valid for edge E if for all cycles containing E, the other 3 edges do not form a "bad" set with C.
        # i.e., if we assign C to E, does any cycle containing E become bad?
        # A cycle (E, E2, E3, E4) becomes bad if colors[E2], colors[E3], colors[E4] are all distinct and distinct from C.
        
        # We can precompute for each edge E, the list of cycles it's in.
        
        def can_assign(e, c):
            for cid in edge_to_cycles[e]:
                c_idx = cycles[cid]
                # Get the other 3 edges
                other_edges = [x for x in c_idx if x != e]
                # Check if they are all assigned and distinct and distinct from c
                # If any is unassigned (-1), it's not bad yet (potential, but not current conflict)
                # Wait, if an edge is unassigned, we can't say it's bad. 
                # So we only reject if it is CURRENTLY bad.
                # But if we assign c, and the other 3 are distinct and != c, then it IS bad.
                # If one of the others is unassigned, we don't know its color yet, so we assume it *could* be valid?
                # No, in backtracking, we check for immediate conflicts with assigned variables.
                # If the other 3 are assigned and distinct and != c, then assigning c creates a bad cycle. Reject.
                
                vals = []
                for oe in other_edges:
                    if colors[oe] == -1:
                        return True # Not a conflict yet
                    vals.append(colors[oe])
                
                # Check if vals has 3 distinct values, all != c
                if len(set(vals)) == 3 and c not in set(vals):
                    return False
                    
            return True

        # The above can_assign is O(deg(e)). deg(e) is number of cycles containing e.
        # Max cycles per edge in K16?
        # An edge (u,v) is in a cycle if we pick 2 other vertices x,y.
        # Number of such pairs is C(14,2) = 91.
        # So we check up to 91 cycles per assignment.
        # 120 edges * 91 checks * 120 colors (worst case) = ~1M ops per path.
        # With backtracking, this might be feasible within 20s if we are lucky or use good heuristics.
        
        # Let's implement the DFS.
        
        def solve_backtrack():
            colors = [-1] * NUM_EDGES
            used_colors = set()
            unassigned = list(range(NUM_EDGES))
            random.shuffle(unassigned)
            
            # We'll use a recursive function with backtracking
            # To avoid recursion depth issues, we might need to increase limit or use iterative.
            # 120 depth is fine for Python.
            
            import sys
            sys.setrecursionlimit(2000)
            
            def dfs(idx):
                if time.time() > time_limit:
                    return False
                
                if idx == len(unassigned):
                    return True
                
                e = unassigned[idx]
                
                # Get candidate colors
                # 1. Unused
                # 2. Used
                
                # To avoid generating full lists, we can iterate.
                # But we need to try them in a specific order.
                
                # Let's create a list of unused colors and shuffle it
                unused_list = [c for c in range(MAX_COLORS) if c not in used_colors]
                random.shuffle(unused_list)
                
                used_list = list(used_colors)
                random.shuffle(used_list)
                
                candidates = unused_list + used_list
                
                for c in candidates:
                    if can_assign(e, c):
                        # Assign
                        colors[e] = c
                        if c not in used_colors:
                            used_colors.add(c)
                        
                        if dfs(idx + 1):
                            return True
                        
                        # Backtrack
                        colors[e] = -1
                        if c in used_colors:
                            # Only remove if no other edge uses it?
                            # We need to check if c is used elsewhere.
                            # This is expensive. 
                            # Better: maintain a count of color usage.
                            pass # We'll fix this with a count array

            # The backtracking with set removal is tricky if a color is used by multiple edges.
            # Let's use a frequency array instead of a set for used_colors.
            
            def solve_backtrack_v2():
                colors = [-1] * NUM_EDGES
                freq = [0] * MAX_COLORS
                unassigned = list(range(NUM_EDGES))
                random.shuffle(unassigned)
                
                def dfs(idx):
                    if time.time() > time_limit:
                        return False
                    
                    if idx == len(unassigned):
                        return True
                    
                    e = unassigned[idx]
                    
                    # Generate candidates
                    # Unused: freq[c] == 0
                    # Used: freq[c] > 0
                    
                    # To prioritize new colors, we pick from unused first.
                    
                    # Optimization: Pre-compute lists? No, they change.
                    
                    # Let's just pick a random color from 0..119, but bias?
                    # No, "New First" is key.
                    
                    # Let's iterate through all 120 colors, but try unused ones first.
                    # We can do this by shuffling the range and partitioning.
                    
                    # Actually, just shuffling 0..119 and trying them in order might be fine if we 
                    # accept that we might try used colors early. 
                    # But "New First" is a specific heuristic.
                    
                    # Let's create two lists: unused and used.
                    unused = [c for c in range(MAX_COLORS) if freq[c] == 0]
                    used = [c for c in range(MAX_COLORS) if freq[c] > 0]
                    
                    random.shuffle(unused)
                    random.shuffle(used)
                    
                    candidates = unused + used
                    
                    for c in candidates:
                        if can_assign(e, c):
                            colors[e] = c
                            freq[c] += 1
                            
                            if dfs(idx + 1):
                                return True
                            
                            freq[c] -= 1
                            colors[e] = -1
                            
                    return False
                
                if dfs(0):
                    return colors
                return None
            
            return solve_backtrack_v2()

    best_colors = None
    best_score = 0
    
    # Run backtracking several times with different seeds/shuffles
    attempts = 0
    while time.time() < deadline - 10.0: # Leave 10s for local search
        attempts += 1
        random.seed(1000 + attempts)
        colors = backtrack_search(time_limit=5.0) # 5s per backtracking attempt
        
        if colors is not None:
            score = len(set(colors))
            if score > best_score:
                best_score = score
                best_colors = colors.copy()
                
            # If we found a high score, we can stop backtracking and go to local search
            if best_score >= 25: # Arbitrary threshold
                break

    # If backtracking failed to find anything, use the recursive bisection as a fallback
    if best_colors is None:
        colors = [0] * NUM_EDGES
        color_id = 0
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        best_colors = colors
        best_score = len(set(colors))

    # --- Phase 2: Local Search Refinement ---
    
    # Use the best_colors found as the starting point for local search.
    current_colors = best_colors
    freq = [0] * MAX_COLORS
    for c in current_colors:
        freq[c] += 1
        
    # Calculate initial bad cycles (should be 0 if valid)
    bad_count = 0
    for c_idx in cycles:
        if is_bad_cycle(current_colors, c_idx):
            bad_count += 1
            
    if bad_count > 0:
        # If backtracking returned an invalid state (shouldn't happen), revert to bisection
        colors = [0] * NUM_EDGES
        color_id = 0
        def process_partition(vertices):
            nonlocal color_id
            if len(vertices) <= 1:
                return
            mid = len(vertices) // 2
            left = vertices[:mid]
            right = vertices[mid:]
            for u in left:
                for v in right:
                    idx = edge_idx[u][v]
                    colors[idx] = color_id
            color_id += 1
            process_partition(left)
            process_partition(right)
        process_partition(list(range(N)))
        current_colors = colors
        freq = [0] * MAX_COLORS
        for c in current_colors:
            freq[c] += 1
        bad_count = 0
        for c_idx in cycles:
            if is_bad_cycle(current_colors, c_idx):
                bad_count += 1

    # Local Search
    while time.time() < deadline:
        # Pick a random edge
        e = random.randint(0, NUM_EDGES - 1)
        old_color = current_colors[e]
        
        # Heuristic: Bias towards unused colors
        if random.random() < 0.7:
            unused = [i for i in range(MAX_COLORS) if freq[i] == 0]
            if unused:
                new_color = random.choice(unused)
            else:
                new_color = random.randint(0, MAX_COLORS - 1)
        else:
            new_color = random.randint(0, MAX_COLORS - 1)

        if new_color == old_color:
            continue

        # Calculate delta in bad_count
        delta = 0
        for cid in edge_to_cycles[e]:
            c_idx = cycles[cid]
            was_bad = is_bad_cycle(current_colors, c_idx)
            
            current_colors[e] = new_color
            is_bad_now = is_bad_cycle(current_colors, c_idx)
            current_colors[e] = old_color
            
            if was_bad and not is_bad_now:
                delta -= 1
            elif not was_bad and is_bad_now:
                delta += 1
            
        new_bad_count = bad_count + delta
        
        if new_bad_count == 0:
            # Accept
            current_colors[e] = new_color
            freq[old_color] -= 1
            freq[new_color] += 1
            bad_count = new_bad_count
            
            score = len(set(current_colors))
            if score > best_score:
                best_score = score
                best_colors = current_colors.copy()

    return best_colors
