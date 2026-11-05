import random
import time
import math
import sys

def construct(n_rows, k):
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Each row has k+1 entries (including 0). 
    # Number of differences per row: k*(k+1)/2
    diffs_per_row = k * (k + 1) // 2
    total_diffs = n_rows * diffs_per_row
    
    # The "scope" is the largest entry.
    # The simple counting bound is total_diffs (since diffs are 1..max_entry and distinct).
    # Actually, max_entry must be at least total_diffs.
    
    # Helper: Check if a set of differences can form a valid row starting at 0
    # A row is defined by its elements a_0=0 < a_1 < ... < a_k.
    # The differences are a_j - a_i.
    # If we assign values to the difference slots, we must ensure they are consistent.
    # Specifically, for any i<j<l, (a_l - a_i) = (a_l - a_j) + (a_j - a_i).
    
    # Strategy:
    # 1. Represent the problem as assigning distinct integers 1..M to all difference slots.
    # 2. The "consistency" constraint is that for each row, the assigned differences must
    #    correspond to a valid set of increasing integers.
    # 3. We use a min-conflicts approach where "conflicts" are:
    #    a) Duplicate difference values (global constraint).
    #    b) Inconsistent difference structures within a row (local constraint).
    
    # To handle row consistency efficiently, we can try to "reconstruct" a row from its assigned differences.
    # If we fix a_0=0, and we know the differences, can we recover the row?
    # Not uniquely from arbitrary differences. But if we assume the differences are "sorted" or have a specific structure?
    # No, the differences are just a set.
    # However, if we pick a "basis" for the row, e.g., the differences involving a_0 (which are just the elements a_1..a_k),
    # then the other differences are determined.
    
    # Alternative View:
    # The row is defined by the elements. The differences are derived.
    # The current program optimizes elements.
    # The proposed strategy optimizes "difference slots".
    
    # Let's define the slots for a row of size k+1:
    # Slot (i, j) corresponds to difference a_j - a_i for 0 <= i < j <= k.
    # There are k*(k+1)/2 slots.
    
    # Consistency Check:
    # Given an assignment of values to slots, is there a row a_0..a_k such that a_j - a_i = val(i,j)?
    # This is equivalent to checking if the graph of differences is "additively consistent".
    # We can pick a_0 = 0.
    # Then a_j = val(0, j).
    # Then we must check that for all i < j, a_j - a_i == val(i, j).
    # So, if we assign values to slots (0, j) for j=1..k, the rest of the row is determined.
    # The "free" variables are the values of slots (0, 1), ..., (0, k).
    # The other slots are "constrained" to be consistent with these.
    
    # So, a "row state" can be represented by the vector of its first k differences: d_j = a_j - a_0 = a_j.
    # Wait, a_j - a_0 is just a_j.
    # So the row is determined by a_1, ..., a_k.
    # The differences are:
    # d_{i,j} = a_j - a_i.
    
    # If we treat the row elements as the variables, we are back to the original formulation.
    # The "difference slot" formulation suggests we treat the *differences* as the primary objects.
    
    # Let's try a different structural change:
    # **Simulated Annealing on the Difference Multiset with Consistency Repair**
    # 1. Start with a random assignment of distinct differences to all slots.
    # 2. This assignment likely violates row consistency.
    # 3. Define a "repair" step: For each row, take the assigned differences and try to find the closest valid row (minimizing L2 distance to assigned diffs, or just checking consistency).
    #    Actually, it's easier to:
    #    - Pick a random row.
    #    - Reconstruct the row from its assigned "anchor" differences (e.g., those involving index 0).
    #    - Update the "implied" differences.
    #    - If the implied differences don't match the assigned ones, we have a conflict.
    
    # This is getting complicated. Let's look at the "Graph Homomorphism" hint.
    # Vertices: Difference slots.
    # Edges: Consistency constraints within a row (e.g., d_{0,2} = d_{0,1} + d_{1,2}).
    # Colors: The integer values 1..M.
    # Constraint: All colors must be distinct globally.
    
    # This is a constraint satisfaction problem (CSP).
    # Min-Conflicts Heuristic for CSPs:
    # 1. Assign a color (value) to each variable (slot).
    # 2. While not solved:
    #    a. Pick a variable involved in a conflict.
    #    b. Assign it the value that minimizes the number of conflicts.
    
    # Conflicts:
    # 1. Global: Two slots have the same value.
    # 2. Local: In a row, the values don't satisfy the additive constraints.
    
    # To make this tractable, we need to define the variables carefully.
    # If we choose variables as the "basis" differences (a_1..a_k for each row), then:
    # - Local consistency is automatically satisfied (we define the row from the basis).
    # - The "derived" differences are just calculated.
    # - The only conflicts are Global (duplicates).
    # This reduces to the original problem: Choose a_1..a_k for each row such that all derived differences are distinct.
    # So the "difference slot" framing doesn't add new power unless we allow "inconsistent" states and repair them.
    
    # Let's try **Iterated Local Search (ILS) with a Strong Perturbation**.
    # The current program uses SA. ILS is often better for combinatorial optimization.
    # Structure:
    # 1. Construct a valid solution (greedy).
    # 2. Local Search to optimize (minimize max entry).
    # 3. Perturb the solution (break validity or just change structure).
    # 4. Local Search again.
    # 5. Repeat.
    
    # The key is the **Perturbation**.
    # Instead of random single moves, we can:
    # - Randomly "shuffle" a subset of rows.
    # - Or, use a "kick" that forces a specific structural change, e.g., swapping two rows' elements or shifting a row.
    
    # Another structural change: **Multi-Objective Optimization via Pareto Front**.
    # Not applicable here, single objective.
    
    # Let's try **Tabu Search**.
    # Tabu Search prevents cycling by keeping a memory of recent moves.
    # This is a structural change from SA (which uses temperature).
    
    # Implementation Plan: Tabu Search
    # 1. State: List of rows.
    # 2. Objective: Max entry.
    # 3. Neighborhood: Change one element in one row.
    # 4. Tabu List: Store (row_idx, col_idx, old_val) for last N moves.
    # 5. Aspiration Criterion: Accept a tabu move if it improves the best known solution.
    # 6. Diversification: If no improvement for X iterations, perform a random "kick" (change several elements randomly) to escape local minima.
    
    # This is distinct from SA. SA uses probabilistic acceptance. Tabu uses deterministic memory.
    
    # Let's implement Tabu Search with Diversification.
    
    def get_max_entry(rows):
        m = 0
        for row in rows:
            if row and row[-1] > m:
                m = row[-1]
        return m

    def is_valid(rows):
        seen = set()
        for row in rows:
            for i in range(len(row)):
                ri = row[i]
                for j in range(i + 1, len(row)):
                    d = row[j] - ri
                    if d in seen:
                        return False
                    seen.add(d)
        return True

    def greedy_construct(seed=None):
        rng = random.Random(seed)
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for pos in range(k):
                found = False
                # Try smallest available
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                    if x > 5000: # Safety
                        break
                if ok:
                    row.append(x)
                    for a in row:
                        used.add(x - a)
                    found = True
                else:
                    # Fallback: just add x
                    row.append(x)
            rows.append(row)
        return rows

    # Initialize
    best_rows = greedy_construct(seed=42)
    if not is_valid(best_rows):
        # Fallback: very safe greedy
        used = set()
        rows = []
        for r in range(n_rows):
            row = [0]
            for pos in range(k):
                x = row[-1] + 1
                while True:
                    ok = True
                    for a in row:
                        if (x - a) in used:
                            ok = False
                            break
                    if ok:
                        break
                    x += 1
                row.append(x)
                for a in row:
                    used.add(x - a)
            rows.append(row)
        best_rows = rows

    best_max = get_max_entry(best_rows)
    
    # Tabu Search Parameters
    tabu_tenure = 100
    max_iter = 500000
    diversify_after = 5000
    
    def tabu_search(initial_rows, deadline):
        current_rows = [row[:] for row in initial_rows]
        curr_max = get_max_entry(current_rows)
        
        # Build diff counts for fast conflict checking
        def build_diff_counts(rows):
            counts = {}
            for row in rows:
                for i in range(len(row)):
                    ri = row[i]
                    for j in range(i + 1, len(row)):
                        d = row[j] - ri
                        counts[d] = counts.get(d, 0) + 1
            return counts

        def count_conflicts(counts):
            conf = 0
            for c in counts.values():
                if c > 1:
                    conf += c - 1
            return conf

        diff_counts = build_diff_counts(current_rows)
        curr_conflicts = count_conflicts(diff_counts)
        
        tabu_list = [] # List of (row_idx, col_idx, old_val, expiry_iter)
        best_local_max = curr_max
        best_local_rows = [row[:] for row in current_rows]
        no_improve_count = 0
        
        for it in range(max_iter):
            if time.time() > deadline:
                break
            
            # Remove expired tabu entries
            while tabu_list and tabu_list[0][3] <= it:
                tabu_list.pop(0)
            
            # Pick a random move
            r_idx = random.randint(0, n_rows - 1)
            c_idx = random.randint(1, k)
            
            row = current_rows[r_idx]
            old_val = row[c_idx]
            
            min_val = row[c_idx - 1] + 1
            max_val = curr_max + 10
            if c_idx < k:
                max_val = min(max_val, row[c_idx + 1] - 1)
            
            if min_val > max_val:
                continue
                
            # Choose new value
            # Bias towards smaller to tighten
            if random.random() < 0.8:
                new_val = random.randint(min_val, min(old_val, max_val))
            else:
                new_val = random.randint(min_val, max_val)
                
            if new_val == old_val:
                continue
            
            # Check if move is tabu
            is_tabu = any(tr == r_idx and tc == c_idx and tv == old_val for tr, tc, tv, te in tabu_list)
            
            # Compute new state
            # Remove old diffs
            old_diffs = []
            for i in range(len(row)):
                if i == c_idx: continue
                d = abs(row[c_idx] - row[i])
                if i < c_idx: d = row[c_idx] - row[i]
                else: d = row[i] - row[c_idx]
                old_diffs.append(d)
                
            for d in old_diffs:
                diff_counts[d] -= 1
                if diff_counts[d] <= 0: del diff_counts[d]
                
            row[c_idx] = new_val
            
            new_diffs = []
            for i in range(len(row)):
                if i == c_idx: continue
                if i < c_idx: d = row[c_idx] - row[i]
                else: d = row[i] - row[c_idx]
                new_diffs.append(d)
                
            for d in new_diffs:
                diff_counts[d] = diff_counts.get(d, 0) + 1
                
            new_conflicts = count_conflicts(diff_counts)
            new_max = max(curr_max, new_val)
            if new_max == curr_max and new_val < old_val:
                # Check if old_val was the unique max
                # Simplification: recompute if it was the max
                if old_val == curr_max:
                    new_max = get_max_entry(current_rows)
            
            cost = new_conflicts * 1000 + new_max
            old_cost = curr_conflicts * 1000 + curr_max
            
            # Aspiration: If new cost is better than best known, accept even if tabu
            aspiration = new_conflicts == 0 and new_max < best_local_max
            
            if not is_tabu or aspiration:
                # Accept move
                curr_max = new_max
                curr_conflicts = new_conflicts
                
                # Add to tabu
                tabu_list.append((r_idx, c_idx, old_val, it + tabu_tenure))
                
                if new_conflicts == 0 and new_max < best_local_max:
                    best_local_max = new_max
                    best_local_rows = [row[:] for row in current_rows]
                    no_improve_count = 0
                else:
                    no_improve_count += 1
            else:
                # Reject: Revert
                row[c_idx] = old_val
                for d in new_diffs:
                    diff_counts[d] -= 1
                    if diff_counts[d] <= 0: del diff_counts[d]
                for d in old_diffs:
                    diff_counts[d] = diff_counts.get(d, 0) + 1
                no_improve_count += 1
            
            # Diversification: If stuck, kick
            if no_improve_count > diversify_after:
                # Randomly change several elements
                for _ in range(5):
                    rr = random.randint(0, n_rows - 1)
                    cc = random.randint(1, k)
                    r = current_rows[rr]
                    old_v = r[cc]
                    min_v = r[cc-1] + 1
                    max_v = curr_max + 50
                    if cc < k: max_v = min(max_v, r[cc+1] - 1)
                    if min_v <= max_v:
                        new_v = random.randint(min_v, max_v)
                        # Apply without tabu check, just update counts
                        old_ds = []
                        for i in range(len(r)):
                            if i == cc: continue
                            d = abs(r[cc] - r[i])
                            if i < cc: d = r[cc] - r[i]
                            else: d = r[i] - r[cc]
                            old_ds.append(d)
                        for d in old_ds:
                            diff_counts[d] -= 1
                            if diff_counts[d] <= 0: del diff_counts[d]
                        
                        r[cc] = new_v
                        new_ds = []
                        for i in range(len(r)):
                            if i == cc: continue
                            if i < cc: d = r[cc] - r[i]
                            else: d = r[i] - r[cc]
                            new_ds.append(d)
                        for d in new_ds:
                            diff_counts[d] = diff_counts.get(d, 0) + 1
                        
                        curr_max = get_max_entry(current_rows)
                        curr_conflicts = count_conflicts(diff_counts)
                
                no_improve_count = 0
                tabu_list = [] # Clear tabu list on kick
                
        return best_local_rows, best_local_max

    # Run Tabu Search with multiple restarts
    current_best_max = best_max
    current_best_rows = best_rows
    
    for restart in range(10):
        if time.time() > deadline - 5:
            break
            
        if restart == 0:
            init_rows = current_best_rows
        else:
            init_rows = greedy_construct(seed=1000 + restart)
            if not is_valid(init_rows):
                # Fix
                used = set()
                rows = []
                for r in range(n_rows):
                    row = [0]
                    for pos in range(k):
                        x = row[-1] + 1
                        while True:
                            ok = True
                            for a in row:
                                if (x - a) in used:
                                    ok = False
                                    break
                            if ok: break
                            x += 1
                        row.append(x)
                        for a in row:
                            used.add(x - a)
                    rows.append(row)
                init_rows = rows
        
        found_rows, found_max = tabu_search(init_rows, deadline - 2)
        
        if is_valid(found_rows) and found_max < current_best_max:
            current_best_max = found_max
            current_best_rows = found_rows

    return current_best_rows
