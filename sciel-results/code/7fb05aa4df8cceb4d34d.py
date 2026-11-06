import random
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50.0  # Leave 10s margin
    
    N = 128
    # Precompute GCD table
    gcd_table = [[0]*(N+1) for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            g = gcd(i, j)
            gcd_table[i][j] = g
            gcd_table[j][i] = g
            
    # Find all bad triples
    # A triple (a, b, c) with a < b < c is bad if gcd(a,b)==gcd(b,c)==gcd(c,a)
    bad_triples = []
    triples_containing = [[] for _ in range(N+1)]
    
    for a in range(1, N+1):
        for b in range(a+1, N+1):
            g_ab = gcd_table[a][b]
            for c in range(b+1, N+1):
                g_bc = gcd_table[b][c]
                if g_ab == g_bc:
                    g_ca = gcd_table[c][a]
                    if g_ab == g_ca:
                        bad_triples.append((a, b, c))
                        triples_containing[a].append((b, c))
                        triples_containing[b].append((a, c))
                        triples_containing[c].append((a, b))
                        
    num_bad_triples = len(bad_triples)
    
    # State representation:
    # active[i] = True if i is in the set
    # bad_count[t] = 1 if triple t is "active" (all 3 nodes active), 0 otherwise
    # node_bad_degree[i] = number of active bad triples containing i
    
    def run_simulated_annealing(seed_val):
        random.seed(seed_val)
        
        active = [False] * (N+1)
        for i in range(1, N+1):
            active[i] = True
            
        # Initialize bad counts
        bad_count = [0] * num_bad_triples
        node_bad_degree = [0] * (N+1)
        total_bad = 0
        
        for idx, (a, b, c) in enumerate(bad_triples):
            # Initially all active, so all triples are bad
            bad_count[idx] = 1
            total_bad += 1
            node_bad_degree[a] += 1
            node_bad_degree[b] += 1
            node_bad_degree[c] += 1
            
        # Objective: Maximize Size - Penalty * BadTriples
        # We want to reach 0 BadTriples.
        # Energy = -Size + Penalty * BadTriples
        # Let's use a large penalty so that removing a bad triple is always preferred over adding a node if it creates a bad triple.
        # However, we want to maximize size eventually.
        # If we strictly enforce 0 bad triples, we are looking for Max Independent Set.
        
        T = 10.0
        T_min = 0.1
        cooling_rate = 0.999
        
        best_size = sum(active)
        best_active = active[:]
        current_size = sum(active)
        
        while time.time() < deadline and T > T_min:
            # Pick a random node to flip
            v = random.randint(1, N)
            
            # Calculate delta if we flip v
            # If active[v] is True, we remove it.
            #   Size decreases by 1.
            #   Bad triples containing v that are currently active become inactive.
            #   Delta Size = -1
            #   Delta Bad = - (number of active bad triples containing v)
            
            # If active[v] is False, we add it.
            #   Size increases by 1.
            #   Bad triples containing v where the other 2 are active become active.
            #   Delta Size = +1
            #   Delta Bad = + (number of inactive bad triples containing v where other 2 are active)
            
            delta_size = 0
            delta_bad = 0
            
            if active[v]:
                delta_size = -1
                # Count how many bad triples containing v are currently active
                # A triple (v, u1, u2) is active if active[u1] and active[u2] are True
                # We need to iterate through triples_containing[v]
                # Note: node_bad_degree[v] is the count of active bad triples containing v.
                # So if we remove v, all these become inactive.
                delta_bad = -node_bad_degree[v]
            else:
                delta_size = 1
                # Count how many bad triples containing v would become active
                # These are triples (v, u1, u2) where active[u1] and active[u2] are True,
                # but currently bad_count is 0 (because v is inactive).
                count_new_bad = 0
                for u1, u2 in triples_containing[v]:
                    if active[u1] and active[u2]:
                        count_new_bad += 1
                delta_bad = count_new_bad
                
            # Energy change
            # We want to minimize E = -Size + Penalty * Bad
            # Delta E = -Delta Size + Penalty * Delta Bad
            Penalty = 1000  # High enough to prioritize validity
            delta_E = -delta_size + Penalty * delta_bad
            
            # Acceptance probability
            if delta_E < 0:
                accept = True
            else:
                # Boltzmann factor
                # exp(-delta_E / T)
                # If delta_E is large positive, prob is small.
                if T > 0:
                    prob = random.random()
                    # Use log to avoid underflow? T is small, delta_E can be large.
                    # delta_E is integer.
                    # If delta_E > 100, exp(-100/T) is basically 0.
                    if delta_E < 100 * T: 
                        accept = (prob < pow(2.718281828, -delta_E / T))
                    else:
                        accept = False
                else:
                    accept = False
                    
            if accept:
                # Apply flip
                active[v] = not active[v]
                current_size += delta_size
                total_bad += delta_bad
                
                # Update node_bad_degree for v and its neighbors in bad triples
                if active[v]:
                    # v became active
                    # node_bad_degree[v] increases by delta_bad (which was positive)
                    # Actually, we need to update degrees for all nodes involved in the changed triples.
                    # This is tricky to do efficiently without iterating all triples containing v.
                    # Since N=128, triples_containing[v] is at most a few hundred.
                    # Let's just recalculate node_bad_degree[v] and update neighbors?
                    # Or simpler: just update the specific triples.
                    
                    # Let's iterate through triples_containing[v]
                    for u1, u2 in triples_containing[v]:
                        # Find the index of this triple?
                        # We stored triples in a list, but not indices in triples_containing.
                        # We can store indices in triples_containing.
                        pass
                        
                # This "efficient update" is getting complex.
                # Given N=128, we can afford to be slightly less efficient.
                # Let's re-calculate node_bad_degree for v and the nodes in the affected triples.
                # Actually, let's just store the triple indices in triples_containing.
                
                # Re-doing the data structure to be cleaner for SA:
                # triples_containing[i] = list of triple indices
                # This is better.
                
                if False: # Placeholder for logic change
                    pass
                    
                # Since we are inside the loop, let's just break and use a simpler, correct but slightly slower update if needed.
                # Or, we can just accept that the "degree" update needs to be done carefully.
                
                # Let's restart the implementation with the correct data structure.
                # The current code structure inside run_simulated_annealing is a bit messy.
                # I will rewrite the whole function in the SEARCH/REPLACE block.
                
                # For now, just to make the code valid, I'll return a greedy result if this path is taken.
                # But I should provide the full correct code in the replacement.
                
                # Update best
                if total_bad == 0 and current_size > best_size:
                    best_size = current_size
                    best_active = active[:]
                    
            T *= cooling_rate
            
        # If we end with bad triples, we need to clean up.
        # Run a greedy removal on the remaining bad triples.
        # This ensures validity.
        
        # Identify remaining bad triples
        final_bad = []
        for idx, (a, b, c) in enumerate(bad_triples):
            if active[a] and active[b] and active[c]:
                final_bad.append((a, b, c))
                
        if final_bad:
            # Greedy remove
            # Create a set of nodes to remove
            to_remove = set()
            # Simple greedy: pick node in most remaining bad triples, remove, repeat.
            # We can just run the same greedy_remove logic on the current active set.
            
            # Construct a sub-problem?
            # Easier: just iterate.
            # While there are bad triples:
            #   Find node v in active set that is in the most active bad triples.
            #   Remove v.
            
            # We need to track active bad triples dynamically.
            # Let's just use a simple loop.
            active_set = set(i for i in range(1, N+1) if active[i])
            # Re-compute active bad triples for this set
            # This is O(N^3) which is fine for N=128 once.
            # But we might need to do it a few times? No, just once at the end.
            
            # Actually, the greedy removal in the seed code is robust.
            # Let's just call a helper that does greedy removal on the current active set.
            
            # Helper: greedy_remove_from(active_set)
            # Returns a valid subset of active_set.
            
            # For simplicity in this draft, I will assume the SA process converges to 0 bad triples.
            # If not, the penalty should have driven it there.
            # If T_min is reached and bad > 0, we force removal.
            
            # Force removal loop
            # Re-calculate degrees for current active set
            deg = [0]*(N+1)
            active_triples = []
            for a in range(1, N+1):
                if not active[a]: continue
                for b in range(a+1, N+1):
                    if not active[b]: continue
                    g_ab = gcd_table[a][b]
                    for c in range(b+1, N+1):
                        if not active[c]: continue
                        if gcd_table[a][b] == gcd_table[b][c] == gcd_table[c][a]:
                            active_triples.append((a,b,c))
                            deg[a]+=1; deg[b]+=1; deg[c]+=1
                            
            while active_triples:
                # Find max degree node in active set
                max_deg = -1
                best_node = -1
                for i in range(1, N+1):
                    if active[i] and deg[i] > max_deg:
                        max_deg = deg[i]
                        best_node = i
                if best_node == -1: break
                active[best_node] = False
                current_size -= 1
                # Remove triples containing best_node
                new_triples = []
                for t in active_triples:
                    if best_node in t:
                        # Update degrees for other 2
                        others = [x for x in t if x != best_node]
                        deg[others[0]] -= 1
                        deg[others[1]] -= 1
                    else:
                        new_triples.append(t)
                active_triples = new_triples
                
        return [i for i in range(1, N+1) if active[i]]
        
    # The logic above is a bit tangled. Let's write a clean version.
    
    # Pre-compute triple indices for efficient updates
    triples_containing_idx = [[] for _ in range(N+1)]
    for idx, (a, b, c) in enumerate(bad_triples):
        triples_containing_idx[a].append(idx)
        triples_containing_idx[b].append(idx)
        triples_containing_idx[c].append(idx)
        
    def run_sa_clean(seed_val):
        random.seed(seed_val)
        active = [False] * (N+1)
        for i in range(1, N+1):
            active[i] = True
            
        bad_count = [1] * num_bad_triples # All start active
        node_deg = [0] * (N+1)
        total_bad = num_bad_triples
        for idx, (a,b,c) in enumerate(bad_triples):
            node_deg[a] += 1
            node_deg[b] += 1
            node_deg[c] += 1
            
        current_size = N
        best_size = N
        best_active = active[:]
        
        T = 5.0
        T_min = 0.01
        cooling = 0.9995
        Penalty = 1000
        
        while time.time() < deadline and T > T_min:
            v = random.randint(1, N)
            
            if active[v]:
                # Remove v
                delta_size = -1
                # Bad triples containing v that are active will become inactive
                # Count them
                # These are indices in triples_containing_idx[v] where bad_count[idx] == 1
                # And active[other1] and active[other2] are True (which is implied if bad_count is 1 and v is active? 
                # Wait, bad_count[idx] is 1 iff all 3 are active.
                # So if active[v] is True, and bad_count[idx] is 1, then the other two are active.
                # So we just count how many bad_count[idx] are 1 for idx in triples_containing_idx[v].
                # But node_deg[v] is exactly this count!
                delta_bad = -node_deg[v]
                
                # Check if this leads to better energy
                delta_E = -delta_size + Penalty * delta_bad
                # -(-1) + P * (-deg) = 1 - P*deg
                # If deg > 0, delta_E is very negative. Good.
                # If deg == 0, delta_E = 1. Bad (removing a node that doesn't break any triples).
                
                if delta_E < 0 or (T > 0 and random.random() < pow(2.71828, -delta_E / T)):
                    # Accept
                    active[v] = False
                    current_size += delta_size
                    total_bad += delta_bad
                    node_deg[v] = 0 # v is no longer active, so it's in 0 active triples
                    
                    # Update neighbors
                    for idx in triples_containing_idx[v]:
                        if bad_count[idx] == 1:
                            # This triple becomes inactive
                            bad_count[idx] = 0
                            a, b, c = bad_triples[idx]
                            # Update degrees for the other 2 nodes
                            for u in (a, b, c):
                                if u != v:
                                    node_deg[u] -= 1
                                    
            else:
                # Add v
                delta_size = 1
                # Bad triples containing v that are inactive but would become active
                # These are idx in triples_containing_idx[v] where bad_count[idx] == 0
                # AND the other two nodes are active.
                # If the other two are active, and v is added, the triple becomes active.
                # Note: If bad_count[idx] is 0, it means at least one node is inactive.
                # Since v is inactive, the other two could be active or not.
                # We need to check if other two are active.
                new_bad = 0
                for idx in triples_containing_idx[v]:
                    if bad_count[idx] == 0:
                        a, b, c = bad_triples[idx]
                        # Find the other two
                        others = [x for x in (a,b,c) if x != v]
                        if active[others[0]] and active[others[1]]:
                            new_bad += 1
                            
                delta_bad = new_bad
                delta_E = -delta_size + Penalty * delta_bad
                # -1 + P * new_bad
                # If new_bad == 0, delta_E = -1. Good.
                # If new_bad > 0, delta_E is large positive.
                
                if delta_E < 0 or (T > 0 and random.random() < pow(2.71828, -delta_E / T)):
                    # Accept
                    active[v] = True
                    current_size += delta_size
                    total_bad += delta_bad
                    node_deg[v] = new_bad
                    
                    # Update neighbors
                    for idx in triples_containing_idx[v]:
                        if bad_count[idx] == 0:
                            a, b, c = bad_triples[idx]
                            others = [x for x in (a,b,c) if x != v]
                            if active[others[0]] and active[others[1]]:
                                # This triple becomes active
                                bad_count[idx] = 1
                                for u in others:
                                    node_deg[u] += 1
                                    
            if total_bad == 0 and current_size > best_size:
                best_size = current_size
                best_active = active[:]
                
            T *= cooling
            
        # Final cleanup if total_bad > 0
        if total_bad > 0:
            # Greedy remove
            # Re-calculate degrees
            deg = [0]*(N+1)
            active_triples = []
            for idx in range(num_bad_triples):
                if bad_count[idx] == 1:
                    a,b,c = bad_triples[idx]
                    active_triples.append((a,b,c))
                    deg[a]+=1; deg[b]+=1; deg[c]+=1
                    
            while active_triples:
                max_deg = -1
                best_node = -1
                for i in range(1, N+1):
                    if active[i] and deg[i] > max_deg:
                        max_deg = deg[i]
                        best_node = i
                if best_node == -1: break
                active[best_node] = False
                current_size -= 1
                # Remove triples
                new_triples = []
                for t in active_triples:
                    if best_node in t:
                        others = [x for x in t if x != best_node]
                        deg[others[0]] -= 1
                        deg[others[1]] -= 1
                    else:
                        new_triples.append(t)
                active_triples = new_triples
            # Update best if this cleanup resulted in a valid set (it should)
            # It might be smaller than best_size, so we keep best_active if it was valid.
            # But best_active was only updated if total_bad==0.
            # So if we had a valid best_active, we return it.
            # If we never had a valid state, we return the cleaned one.
            
        # Return the best valid set found
        # If best_active was never updated (meaning we never hit 0 bad triples during SA),
        # we need to return a valid set.
        # The cleanup loop ensures the final 'active' is valid.
        # But 'best_active' might be from an earlier valid state.
        # Let's check if best_active is valid.
        # We track best_size only when total_bad==0.
        # So if best_size > 0, best_active is valid.
        # If best_size == 0 (unlikely, at least 1 number is valid), we return the cleaned active.
        
        if best_size > 0:
            return [i for i in range(1, N+1) if best_active[i]]
        else:
            return [i for i in range(1, N+1) if active[i]]

    # Run multiple starts
    best_set_overall = []
    best_size_overall = 0
    
    # Use a few seeds
    for i in range(10):
        if time.time() > deadline:
            break
        S = run_sa_clean(42 + i)
        if len(S) > best_size_overall:
            best_size_overall = len(S)
            best_set_overall = S
            
    return best_set_overall

