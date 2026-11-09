import time
import random
from math import gcd

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin before 90s kill
    
    n = 128
    nums = list(range(1, n + 1))
    
    # 1. Precompute GCDs
    gcds = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g = gcd(nums[i], nums[j])
            gcds[i][j] = g
            gcds[j][i] = g
            
    # 2. Identify "Bad Pairs"
    # A pair (i, j) is "bad" if there exists k != i, j such that 
    # gcd(i,j) == gcd(i,k) == gcd(j,k).
    # If we select both i and j, we CANNOT select any k that completes this triple.
    # More importantly for MIS formulation:
    # We want a set S with no bad triples.
    # Let's define a graph G where vertices are numbers 1..128.
    # An edge (u, v) exists if {u, v} is part of at least one bad triple.
    # Wait, that's not quite right for MIS.
    
    # Let's re-evaluate the constraint.
    # A set S is valid if there are NO three distinct i, j, k in S such that 
    # gcd(i,j) == gcd(i,k) == gcd(j,k).
    
    # This is a 3-uniform hypergraph independent set problem.
    # Vertices: 1..128.
    # Hyperedges: Triples (i,j,k) that are "bad".
    # We want the maximum size subset of vertices containing no hyperedge.
    
    # Converting 3-uniform hypergraph MIS to Graph MIS is not direct.
    # However, we can use a different perspective or a specific solver.
    
    # Alternative: Branch and Bound directly on the hypergraph.
    # Or, we can transform it.
    # If we pick two elements u, v, we must exclude all w such that (u,v,w) is bad.
    # This suggests a recursive search where we pick an element, and then we must
    # remove all elements that form a bad triple with any pair of already picked elements?
    # No, that's not right either. We pick elements one by one.
    # If we pick u, then v, then we cannot pick any w where (u,v,w) is bad.
    # But we also cannot pick any x such that (u,x,y) is bad for some y we might pick later?
    # The constraint is global.
    
    # Let's stick to the hypergraph MIS formulation and use a specialized B&B.
    # State: (candidate_set, current_set_size, forbidden_set)
    # This is complex.
    
    # Let's try the Graph MIS approach with a "conflict" definition.
    # Define graph G where edge (u, v) exists if u and v "conflict".
    # Do u and v conflict? 
    # If we include u and v in our set S, we are forbidden from including ANY w 
    # such that {u, v, w} is a bad triple.
    # This doesn't mean u and v cannot be in S together. They CAN be, as long as 
    # we don't pick the specific w's.
    # So a simple graph MIS on "conflict pairs" is NOT equivalent.
    
    # However, consider the structure of bad triples.
    # A bad triple (a,b,c) requires gcd(a,b)=gcd(a,c)=gcd(b,c)=d.
    # This implies a,b,c are all multiples of d.
    # Let a=da', b=db', c=dc'. Then gcd(a',b')=gcd(a',c')=gcd(b',c')=1.
    # So a', b', c' are pairwise coprime.
    
    # Strategy:
    # We can decompose the problem by the GCD value d.
    # For each d, let S_d be the set of numbers in [1,128] divisible by d.
    # If we pick 3 numbers from S_d, say a,b,c, they form a bad triple IF 
    # gcd(a/d, b/d) == gcd(a/d, c/d) == gcd(b/d, c/d) == 1.
    
    # This is still complex. Let's use a robust heuristic search with strong pruning.
    # Given the small N=128, a well-tuned Branch and Bound on the hypergraph 
    # with greedy lower/upper bounds might work.
    
    # Upper Bound: 
    # If we have a set of candidates C, and we have already selected K elements.
    # The maximum number of elements we can add is at most |C|.
    # Can we get a tighter bound?
    # If we pick an element x from C, we must remove from C all elements y such that 
    # there exists z in C (z != x, y) where (x,y,z) is bad? No, z doesn't have to be in C yet.
    # Actually, if we pick x and y, we must exclude any z such that (x,y,z) is bad.
    # So if we commit to x, we don't immediately remove others, but the space of valid 
    # completions shrinks.
    
    # Let's implement a recursive search:
    # search(candidates, current_size, forbidden_pairs)
    # This state is too large.
    
    # Let's try a different angle: Local Search with Simulated Annealing is often 
    # better for these "constraint satisfaction" type maximization problems on small graphs 
    # than exact B&B which might explode.
    # The current program uses Greedy + Random Restart. 
    # Inspiration 3 used Simulated Annealing.
    # Inspiration 2 used MIS (which I argued is not directly applicable without transformation).
    
    # Let's try a "Greedy + Local Search" with a more sophisticated move set.
    # Moves:
    # 1. Add a valid element.
    # 2. Remove an element.
    # 3. Swap an element (remove one, add another).
    
    # Key Insight for Speed: 
    # Precompute for each pair (i,j) the set of "killers" K(i,j) = {k | (i,j,k) is bad}.
    # If i and j are in S, then no k in K(i,j) can be in S.
    
    # We can maintain a "forbidden mask" for each potential element x.
    # forbidden[x] = set of elements y in S such that there exists z in S with (x,y,z) bad?
    # No, if x is NOT in S, we don't care.
    # If we want to add x to S, we must check:
    # For all pairs (y,z) in S, is x in K(y,z)?
    # This is O(|S|^2) per addition check.
    
    # Optimization:
    # Maintain a count for each x: conflict_count[x] = number of pairs (y,z) in S such that x in K(y,z).
    # If conflict_count[x] == 0, x can be added.
    # When we add a new element w to S:
    #   For every other element v in S:
    #       For every x in K(v,w):
    #           conflict_count[x] += 1
    #   When we remove an element w from S:
    #       For every other element v in S:
    #           For every x in K(v,w):
    #               conflict_count[x] -= 1
    
    # This allows O(1) check for "can add x" (just check conflict_count[x]==0).
    # Update cost when adding w: Sum of |K(v,w)| for v in S.
    # How big is K(v,w)?
    # K(v,w) are numbers k such that gcd(v,w)=gcd(v,k)=gcd(w,k).
    # Let d = gcd(v,w). Then v=da, w=db, gcd(a,b)=1.
    # k must be multiple of d. k=dc.
    # gcd(a,c)=1 and gcd(b,c)=1.
    # So c must be coprime to a*b.
    # Number of such c in [1, 128/d] is roughly (128/d) * phi(a*b)/(a*b).
    # Since a,b are coprime, phi(a*b)=phi(a)phi(b).
    # This is usually small.
    
    # Let's implement this efficient Local Search.
    
    # Precompute K[i][j] as a list of indices k (0-indexed) such that {i,j,k} is bad.
    K = [[[] for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            g_ij = gcds[i][j]
            killers = []
            for k in range(n):
                if k == i or k == j:
                    continue
                if gcds[i][k] == g_ij and gcds[j][k] == g_ij:
                    killers.append(k)
            K[i][j] = killers
            K[j][i] = killers

    # Initial Solution: Greedy
    def greedy_init():
        S = []
        S_set = set()
        # Order by degree in conflict graph? Or random?
        # Let's try ordering by number of "conflicts" (size of union of K(i,j)?)
        # Or just random restarts.
        
        # To make it fast, we'll do a few greedy passes with different orders.
        best_S = []
        
        for restart in range(100):
            if time.time() > deadline:
                break
            order = list(range(n))
            random.shuffle(order)
            
            S = []
            S_set = set()
            conflict_counts = [0] * n
            
            for x in order:
                if conflict_counts[x] == 0:
                    # Add x
                    S.append(x)
                    S_set.add(x)
                    # Update conflict counts for all other potential additions
                    # For each v in S (excluding x just added? No, x is now in S)
                    # We need to update counts for all y NOT in S? 
                    # Actually, conflict_counts[y] tracks how many pairs in S "kill" y.
                    # When we add x, new pairs are (x, v) for all v in S (before adding x).
                    for v in S: # v is in S, x is being added
                         # Pairs (x, v) are formed.
                         # For each k in K[x][v], conflict_counts[k] increases by 1.
                         for k in K[x][v]:
                             if k not in S_set: # Only care about elements not in S? 
                                 # Actually, if k is in S, it's already "in". 
                                 # But if k in S, then (x,v,k) is a bad triple in S, which is invalid.
                                 # Wait, if k is in S, and we add x, and v is in S, 
                                 # and k in K[x][v], then S becomes invalid.
                                 # So we must ensure that NO k in K[x][v] is in S.
                                 # My "conflict_counts" logic assumed we only add if count is 0.
                                 # If count is 0, it means no pair in S kills x.
                                 # This implies x is not in K[v][w] for any v,w in S.
                                 # This does NOT imply that adding x doesn't kill some existing element?
                                 # No, "killing" is about forming a bad triple.
                                 # If x is added, and v,w are in S, and x in K[v][w], then (v,w,x) is bad.
                                 # So we check if x is in K[v][w] for any v,w in S.
                                 # This is equivalent to checking if conflict_counts[x] > 0.
                                 # If conflict_counts[x] == 0, then x is safe to add.
                                 # When we add x, does it affect the validity of S?
                                 # No, because we checked x is safe.
                                 # Does it affect the "safety" of other elements y not in S?
                                 # Yes. The new pairs involving x are (x, v) for v in S.
                                 # Any y in K[x][v] is now "killed" by the pair (x,v).
                                 # So we must increment conflict_counts[y] for all y in K[x][v].
                                 pass
                         # Optimization: Only update y if y is not in S? 
                         # If y is in S, and y in K[x][v], then (x,v,y) is bad.
                         # But we already ensured x is safe, meaning x is not in K[v][w] for any v,w in S.
                         # This is symmetric. If y in K[x][v], then x in K[y][v].
                         # Since y,v are in S, if x in K[y][v], then conflict_counts[x] would have been > 0.
                         # So if conflict_counts[x] == 0, then x is NOT in K[y][v] for any y,v in S.
                         # Thus, adding x does NOT create a bad triple with existing elements.
                         # So we only need to update conflict_counts for y NOT in S? 
                         # Actually, we can update all, but checking "if k not in S_set" saves time.
                         for k in K[x][v]:
                             if k not in S_set:
                                 conflict_counts[k] += 1
                            
            if len(S) > len(best_S):
                best_S = S[:]
                
        return best_S

    # Local Search: Simulated Annealing
    def local_search(initial_S, time_limit):
        S = list(initial_S)
        S_set = set(S)
        conflict_counts = [0] * n
        
        # Initialize conflict_counts
        # For each pair (u,v) in S, increment conflict_counts[k] for k in K[u][v] if k not in S
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                u = S[i]
                v = S[j]
                for k in K[u][v]:
                    if k not in S_set:
                        conflict_counts[k] += 1
        
        best_S = S[:]
        best_size = len(S)
        
        current_temp = 1.0
        cool_rate = 0.95
        rng = random.Random(123)
        
        while time.time() < time_limit:
            if current_temp < 0.01:
                break
            
            # Generate a move
            # 1. Add a random valid element
            # 2. Remove a random element
            # 3. Swap: Remove one, Add one
            
            move_type = rng.random()
            
            if move_type < 0.5:
                # Try to add
                # Pick random candidate not in S
                candidates = [i for i in range(n) if i not in S_set and conflict_counts[i] == 0]
                if not candidates:
                    # Try to remove to free space?
                    # For now, skip or try remove
                    continue
                x = rng.choice(candidates)
                # Add x
                S.append(x)
                S_set.add(x)
                # Update counts
                for v in S:
                    if v == x: continue
                    for k in K[x][v]:
                        if k not in S_set:
                            conflict_counts[k] += 1
                delta_size = 1
                # Check if valid (should be by construction)
                
            elif move_type < 0.8:
                # Try to remove
                if not S:
                    continue
                x = rng.choice(S)
                S.remove(x)
                S_set.remove(x)
                # Update counts: decrement for pairs (x,v)
                for v in S:
                    for k in K[x][v]:
                        if k not in S_set:
                            conflict_counts[k] -= 1
                delta_size = -1
                
            else:
                # Swap: Remove x, Add y
                if not S:
                    continue
                x = rng.choice(S)
                # Find y not in S such that if we remove x, y becomes valid?
                # Or just pick random y not in S.
                # We need to check if y is valid AFTER removing x.
                # This is tricky. 
                # Let's just try: Remove x, then try to add a random valid y.
                # If y is valid, we did a swap. If not, we just removed x (size -1).
                # To make it a true swap, we should look for y that becomes valid.
                
                # Temporarily remove x
                S.remove(x)
                S_set.remove(x)
                # Update counts for removal of x
                for v in S:
                    for k in K[x][v]:
                        if k not in S_set:
                            conflict_counts[k] -= 1
                
                # Now find a valid y
                candidates = [i for i in range(n) if i not in S_set and conflict_counts[i] == 0]
                if candidates:
                    y = rng.choice(candidates)
                    # Add y
                    S.append(y)
                    S_set.add(y)
                    for v in S:
                        if v == y: continue
                        for k in K[y][v]:
                            if k not in S_set:
                                conflict_counts[k] += 1
                    delta_size = 0 # Removed 1, Added 1
                else:
                    # No valid y to add, so we just removed x
                    delta_size = -1
                    # We might want to revert if we only want swaps? 
                    # But SA allows size changes.
                    pass
            
            # Evaluate
            new_size = len(S)
            # In this problem, "cost" is just -size. We want to maximize size.
            # So delta_cost = -delta_size.
            # Accept if delta_cost < 0 (improvement) or with prob exp(-delta_cost/T)
            
            if delta_size > 0:
                # Improvement
                if new_size > best_size:
                    best_size = new_size
                    best_S = S[:]
                # Always accept improvement
                pass
            else:
                # Deterioration or neutral
                delta_cost = -delta_size
                if delta_cost == 0:
                    # Neutral, accept?
                    pass
                else:
                    prob = pow(2.718281828, -delta_cost / current_temp)
                    if rng.random() > prob:
                        # Reject move: Revert
                        # This requires storing state or reverting operations.
                        # For simplicity in this sketch, I'll implement a simple revert logic 
                        # by keeping a stack or just re-initializing from best if rejected?
                        # Re-initializing is slow.
                        # Let's implement proper revert.
                        
                        # Since I didn't store old state, I need to reverse the operation.
                        # This is getting complex for a single pass.
                        # Let's use a simpler "Hill Climbing with Random Restarts" 
                        # or a very simple SA that keeps track of state.
                        
                        # Given time constraints, let's stick to Greedy + Random Restarts 
                        # but with the efficient conflict count update.
                        # The current program does this but maybe not efficiently enough 
                        # or with enough restarts.
                        
                        # I will implement the efficient Greedy + Local Search (Add/Remove) 
                        # without complex SA revert logic, just simple acceptance of improvements 
                        # and random kicks.
                        
                        # Actually, let's just do the Greedy with the efficient update 
                        # and many restarts. The bottleneck in the original was O(N^2) check.
                        # This is O(1) check + O(K) update.
                        
                        pass # Reject logic placeholder
            
            current_temp *= cool_rate
            
        return best_S

    # Run Greedy Init
    S0 = greedy_init()
    
    # Run Local Search from S0
    # I need a working Local Search. Let's implement a simple "First Improvement" local search.
    def local_search_simple(S_init, time_limit):
        S = list(S_init)
        S_set = set(S)
        conflict_counts = [0] * n
        
        # Init counts
        for i in range(len(S)):
            for j in range(i+1, len(S)):
                u = S[i]
                v = S[j]
                for k in K[u][v]:
                    if k not in S_set:
                        conflict_counts[k] += 1
        
        best_S = S[:]
        best_size = len(S)
        
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            # Try to add any valid element
            for x in range(n):
                if x not in S_set and conflict_counts[x] == 0:
                    # Add x
                    S.append(x)
                    S_set.add(x)
                    for v in S:
                        if v == x: continue
                        for k in K[x][v]:
                            if k not in S_set:
                                conflict_counts[k] += 1
                    improved = True
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S[:]
                    break # Restart scan? Or continue?
            
            # If no addition, try to remove one and add another?
            # Simple removal to break deadlock?
            # If stuck, remove a random element and try again.
            if not improved and S:
                # Remove random element
                x = random.choice(S)
                S.remove(x)
                S_set.remove(x)
                for v in S:
                    for k in K[x][v]:
                        if k not in S_set:
                            conflict_counts[k] -= 1
                improved = True # Force another pass
        
        return best_S

    best_S = S0
    # Run local search from S0
    if time.time() < deadline:
        best_S = local_search_simple(S0, deadline)
        
    # Try a few more random restarts if time permits
    for restart in range(20):
        if time.time() > deadline:
            break
        S_r = greedy_init()
        S_r = local_search_simple(S_r, deadline)
        if len(S_r) > len(best_S):
            best_S = S_r
            
    if not best_S:
        return [1]
        
    result = sorted(nums[i] for i in best_S)
    return result

# EVOLVE-BLOCK-END
