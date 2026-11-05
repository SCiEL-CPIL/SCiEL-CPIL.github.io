import time
import random
import sys
from itertools import combinations

def construct(n_rows, k):
    """
    Construct n_rows rows of k+1 increasing integers starting at 0
    such that all positive differences within rows are distinct globally.
    Minimize the largest entry (scope).
    
    Strategy: Maximum Clique Search on Conflict Graph.
    1. For a given scope M, generate a set of candidate rows.
       A row is valid if its internal differences are distinct.
    2. Build a conflict graph where an edge exists between two rows if their 
       difference sets are disjoint.
    3. Use a branch-and-bound algorithm (Bron-Kerbosch variant with pivoting) 
       to find a clique of size n_rows.
    4. If found, return the rows. If not, increase M.
    
    To keep the graph manageable, we limit the number of candidate rows 
    by sampling or restricting to "structural" rows if M is large.
    For (7,6), M is around 112. Number of rows C(113, 7) is huge (~10^9).
    So we cannot generate all rows. 
    Instead, we use a heuristic generation of candidate rows:
    - Generate random valid rows.
    - Or, generate rows based on "sparse" structures.
    
    Actually, for small M (like 105-115), the number of valid rows is still large.
    However, we can use a randomized approach to build the graph incrementally 
    or use a different search: 
    "Randomized Backtracking with Conflict Pruning" is essentially what the seed does.
    
    Let's try a different angle: **Iterative Deepening with Bitset Conflict Checking**.
    We pre-compute the set of all differences for each candidate row as a bitset.
    We maintain a bitset of available differences.
    
    Wait, the directive says: "Formulate the problem as a maximum independent set search in a conflict graph... using a branch-and-bound solver with bitset acceleration".
    
    Since generating all C(M+1, k+1) rows is impossible for M=112, k=6, 
    we must generate a *subset* of promising rows.
    
    Promising rows are those that use "rare" differences or have specific structures.
    Alternatively, we can use a **Randomized Local Search** on the set of rows directly, 
    but the prompt asks for the Clique formulation.
    
    Let's implement a **Hybrid**:
    1. Generate a pool of candidate rows (e.g., 5000 random valid rows).
    2. Build the compatibility graph (edge if disjoint diffs).
    3. Run Max Clique (Bron-Kerbosch) to find a clique of size n_rows.
    4. If found, done.
    5. If not, add more random rows or increase M.
    
    This is structurally different from greedy row-by-row because it considers the global compatibility of the *set* of rows, allowing it to backtrack and swap rows in a way that greedy cannot.
    """
    start_time = time.time()
    DEADLINE = 145.0

    num_elements = k + 1
    diffs_per_row = num_elements * (num_elements - 1) // 2

    # Lower bound
    total_diffs_needed = n_rows * diffs_per_row
    lower_bound = total_diffs_needed

    def is_timeup():
        return time.time() > DEADLINE

    def get_diffs(row):
        diffs = set()
        for i in range(len(row)):
            for j in range(i+1, len(row)):
                diffs.add(row[j] - row[i])
        return diffs

    def generate_random_valid_row(M):
        """
        Generate a random row of length num_elements in [0, M] 
        such that internal differences are distinct.
        """
        # Simple rejection sampling might be slow if M is small and k is large,
        # but for k=6, M=112, it's likely fast enough.
        while True:
            if is_timeup():
                return None
            row = [0] + sorted(random.sample(range(1, M+1), num_elements-1))
            diffs = get_diffs(row)
            if len(diffs) == diffs_per_row:
                return row

    def solve_with_clique_search(M, num_candidates):
        """
        Try to find a clique of size n_rows in the compatibility graph
        of 'num_candidates' random valid rows with max entry <= M.
        """
        candidates = []
        diff_sets = []
        
        # Generate candidates
        attempts = 0
        while len(candidates) < num_candidates and attempts < num_candidates * 10:
            if is_timeup():
                return None
            row = generate_random_valid_row(M)
            if row is not None:
                diffs = get_diffs(row)
                # Check for duplicates in candidates?
                # If a row is identical to an existing one, it has the same diffs, 
                # so it's not useful for a clique of distinct rows? 
                # Actually, rows must be distinct? 
                # The task says "Choose 7 rows...". It doesn't explicitly say distinct rows,
                # but if two rows are identical, their diffs are identical, so they conflict.
                # So we need distinct diff sets.
                if diffs not in diff_sets:
                    candidates.append(row)
                    diff_sets.append(diffs)
            attempts += 1
        
        if len(candidates) < n_rows:
            return None

        # Build adjacency list
        # adj[i] is a set of j such that candidates[i] and candidates[j] are compatible (disjoint diffs)
        n = len(candidates)
        adj = [set() for _ in range(n)]
        
        # Optimization: Use bitsets for fast intersection checks?
        # For n=5000, O(n^2) is 25M ops. In Python, set intersection is fast but 25M * (cost of set op) might be slow.
        # Let's try to limit candidates to 1000-2000 to keep it fast.
        
        for i in range(n):
            di = diff_sets[i]
            for j in range(i+1, n):
                if is_timeup():
                    return None
                dj = diff_sets[j]
                if not di & dj:
                    adj[i].add(j)
                    adj[j].add(i)

        # Bron-Kerbosch with Pivoting for Maximum Clique
        # We want a clique of size n_rows.
        
        best_clique = []
        
        def bron_kerbosch(R, P, X):
            nonlocal best_clique
            if is_timeup():
                return
            if len(R) > len(best_clique):
                best_clique = list(R)
            if len(best_clique) >= n_rows:
                return # Found solution
                
            if not P and not X:
                return

            # Pivot selection: choose u in P ∪ X with maximum degree in P
            # Heuristic: pick u with max |P ∩ adj[u]|
            U = P.union(X)
            if not U:
                return
            
            # Find pivot u
            max_deg = -1
            u = None
            # Randomize pivot selection to escape local structures?
            # Standard Bron-Kerbosch picks max degree.
            for v in U:
                deg = len(P.intersection(adj[v]))
                if deg > max_deg:
                    max_deg = deg
                    u = v
            
            # Iterate over P \ N(u)
            # P \ N(u) are vertices in P that are NOT adjacent to u
            # In a clique search, we branch on vertices in P.
            # To reduce branches, we only branch on vertices NOT adjacent to u.
            # Why? Because any clique containing u must also contain all other vertices in the clique that are in P.
            # If we branch on a vertex v not adjacent to u, we cover cliques that don't contain u.
            # Cliques that DO contain u are covered when we branch on u (if u is in P \ N(u)? No, u is in U, not necessarily P).
            # Wait, standard BK:
            # If P is empty, return.
            # Choose pivot u in P ∪ X.
            # For each v in P \ N(u):
            #   BK(R ∪ {v}, P ∩ N(v), X ∩ N(v))
            #   P = P \ {v}
            #   X = X ∪ {v}
            
            # We need P \ N(u).
            # N(u) is adj[u].
            # So we iterate v in P such that v not in adj[u].
            
            # To improve performance, we can order vertices by degree or randomly.
            # Let's try random order for diversity.
            P_list = list(P)
            random.shuffle(P_list)
            
            for v in P_list:
                if v in adj[u]:
                    continue
                # Branch
                R.add(v)
                P_new = P.intersection(adj[v])
                X_new = X.intersection(adj[v])
                bron_kerbosch(R, P_new, X_new)
                R.remove(v)
                P.remove(v)
                X.add(v)
                
                if len(best_clique) >= n_rows:
                    return

        # Initial call
        # R = empty set
        # P = all vertices
        # X = empty set
        P_init = set(range(n))
        X_init = set()
        R_init = set()
        
        # To help find a clique of size n_rows quickly, we can use a heuristic:
        # Start with a greedy clique and then refine?
        # Or just run BK.
        
        # BK can be exponential. We need a time limit inside.
        # The is_timeup check handles this.
        
        # Optimization: If n_rows is small (7), we can stop as soon as we find one.
        
        bron_kerbosch(R_init, P_init, X_init)
        
        if len(best_clique) >= n_rows:
            return [candidates[i] for i in best_clique[:n_rows]]
        return None

    # Main loop
    # Try M from lower_bound upwards
    # Increase number of candidates as M increases or if previous fails?
    # For M=112, the solution is known to exist.
    # We need enough candidates to include a valid set.
    
    best_solution = None
    
    # Start from lower bound, but maybe jump to known best if lower bound is far?
    # Lower bound for (7,6) is 7 * 21 = 147? 
    # Wait. diffs_per_row for k=6 (7 elements) is 7*6/2 = 21.
    # Total diffs = 7 * 21 = 147.
    # So M >= 147?
    # The task says "best known set has largest entry 112".
    # How is that possible?
    # "all differences between two entries of the same row are distinct over all rows"
    # Total number of differences used is n_rows * diffs_per_row.
    # These differences must be distinct integers from 1 to M.
    # So M must be at least the total count of differences.
    # 147 > 112.
    # There is a contradiction in my understanding or the task description.
    
    # Let's re-read: "Difference triangle set (7,5)"
    # k=5? "Choose 7 rows of 6 increasing integers".
    # If k=5, then num_elements = 6.
    # diffs_per_row = 6*5/2 = 15.
    # Total diffs = 7 * 15 = 105.
    # Lower bound M >= 105.
    # Best known 112.
    # This matches!
    # So k in the task title (7,5) likely means n_rows=7, k=5.
    # The function signature is construct(n_rows, k).
    # The prompt says: "Choose 7 rows of 6 increasing integers".
    # This implies n_rows=7, k=5 (since 6 integers = k+1).
    # The evaluator passes n_rows and k.
    # If the evaluator passes k=5, then num_elements=6.
    # If the evaluator passes k=6, then num_elements=7.
    # The prompt text says "(7,5)" in the title, but the description says "6 increasing integers".
    # I will assume the evaluator passes k such that k+1 is the number of integers.
    # Based on "6 increasing integers", k=5.
    
    # My code uses num_elements = k + 1. This is correct.
    
    # Let's verify the lower bound logic.
    # If k=5, diffs_per_row=15, total=105. LB=105.
    # If k=6, diffs_per_row=21, total=147. LB=147.
    
    # The prompt says "The best known set has largest entry 112... simple counting bound is 105".
    # This confirms k=5 case.
    
    # So if the evaluator passes k=5, we search M from 105.
    # If k=6, we search M from 147.
    
    # Number of candidates:
    # For M=112, k=5 (6 elements), C(113, 6) is ~ 2 * 10^9.
    # We can't generate all.
    # We need to generate a subset that likely contains a solution.
    # 1000-5000 candidates might be enough for random sampling if the solution density is high.
    # However, if the solution is sparse, we might miss it.
    
    # Strategy:
    # 1. Start with M = lower_bound.
    # 2. Generate N candidates (e.g., 2000).
    # 3. Run Clique Search.
    # 4. If fail, increase N or M.
    
    # To be safe, we'll increase M slowly and keep N constant initially, then increase N if time permits.
    
    # Note: Bron-Kerbosch on 2000 nodes with average degree?
    # If diffs are random, probability of disjoint is high?
    # Diffs are in [1, M]. Row uses 15 diffs.
    # P(conflict) is high if M is close to 105.
    # Graph is likely sparse or dense?
    # If M=105, total diffs 105. Each row uses 15.
    # Two random rows: expected overlap?
    # This is a set packing problem.
    
    # Let's try.
    
    M = lower_bound
    max_M = lower_bound + 20  # Search a bit above LB
    
    while M <= max_M and not is_timeup():
        # Try with a few different candidate pool sizes
        for num_cands in [1000, 2000, 3000]:
            if is_timeup():
                break
            
            res = solve_with_clique_search(M, num_cands)
            if res is not None:
                # Verify
                all_diffs = []
                valid = True
                for r in res:
                    if r is None:
                        valid = False
                        break
                    for i in range(len(r)):
                        for j in range(i+1, len(r)):
                            all_diffs.append(r[j] - r[i])
                
                if len(all_diffs) == len(set(all_diffs)) and len(all_diffs) == n_rows * diffs_per_row:
                    best_solution = res
                    break
        if best_solution is not None:
            break
        M += 1
        
    if best_solution is not None:
        return best_solution

    # Fallback: Greedy construction (from seed)
    # This ensures we return *something* valid, even if suboptimal.
    used = set()
    rows = []
    for _ in range(n_rows):
        row = [0]
        x = 0
        while len(row) < num_elements:
            if is_timeup():
                break
            x += 1
            new_diffs = {x - a for a in row}
            if not new_diffs & used:
                row.append(x)
                used |= new_diffs
        if len(row) == num_elements:
            rows.append(row)
        else:
            # If we ran out of time or space, fill with large numbers to make it valid?
            # No, must be valid.
            # Just return what we have? No, must be n_rows.
            # This fallback is risky if time is up.
            # But the clique search should have found something for (7,5).
            pass
            
    if len(rows) == n_rows:
        return rows
        
    # Absolute fallback: return a trivially valid set with large scope
    # This scores poorly but is valid.
    # We construct rows with huge gaps.
    rows = []
    base = 1
    for i in range(n_rows):
        row = [0]
        curr = 1
        for j in range(1, num_elements):
            curr += 1000 * (i+1) * (j+1)
            row.append(curr)
        rows.append(row)
    return rows

