import time
import random
from math import gcd

def construct():
    t0 = time.time()
    DEADLINE = 85.0  # Leave 5s margin for the 90s kill switch
    N = 128
    UNIVERSE = list(range(1, N + 1))
    
    # Precompute GCDs
    gcd_mat = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i + 1, N + 1):
            g = gcd(i, j)
            gcd_mat[i][j] = g
            gcd_mat[j][i] = g

    # Identify all bad triples
    # A triple (a, b, c) with a < b < c is bad if gcd(a,b) == gcd(a,c) == gcd(b,c)
    bad_triples = []
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            g_ab = gcd_mat[a][b]
            for c in range(b + 1, N + 1):
                if gcd_mat[a][c] == g_ab and gcd_mat[b][c] == g_ab:
                    bad_triples.append((a, b, c))
    
    # Build conflict structure
    # For each element, find all elements that cannot be in the solution with it
    # Actually, the constraint is on triples. 
    # Let's use a branch and bound approach.
    
    # To make B&B efficient, we can use a greedy coloring bound on the hypergraph?
    # Or simply use a simple bound: 
    # If we include v, we must exclude all u such that there exists w with (v,u,w) bad.
    # Let's precompute for each v, the set of "neighbors" that conflict with v in some triple.
    # Wait, if (v, u, w) is bad, then v, u, w cannot all be in the set.
    # This is a 3-uniform hypergraph.
    
    # Let's try a different approach: 
    # Greedy construction + Local Search (1-swap/1-add)
    # This was inspiration 3 (score 1.5).
    # But the prompt asks for a DIFFERENT strategy from the current program (which is pruning).
    # The current program is "prune from full".
    # Inspiration 1 & 2 are B&B.
    # Inspiration 3 is Greedy + Local Search.
    
    # Let's implement a robust Greedy + Local Search, which is distinct from the current Pruning approach.
    # Or, since Inspirations 1 & 2 scored 1.625, let's try to implement a simplified B&B or a very strong Greedy+LS.
    # Given the time limit, a well-tuned Greedy+LS with multiple restarts is often safer than B&B which can blow up.
    # However, the prompt says "Model the constraint as a 2-SAT-like conflict structure...".
    # This suggests looking at the structure of conflicts.
    
    # Let's stick to the directive: "KEEP this design and investigate improvements within its family."
    # The current design is "Prune from full".
    # So I should improve the Pruning approach.
    # The current pruning is O(M^2) in the worst case per step because it scans all triples.
    # I can improve this by maintaining a count of active triples for each element.
    
    # Refined Strategy: 
    # 1. Precompute for each element v, the list of bad triples it belongs to.
    # 2. Maintain a count of active triples for each element.
    # 3. When removing an element, decrement counts for its neighbors in those triples.
    # 4. Use a priority queue to always pick the element with the highest "impact" (most active triples).
    # 5. This is a more efficient version of the current "Most Central" removal.
    
    # Let's implement this efficient pruning.
    
    # Precompute incidence
    triple_incidence = [[] for _ in range(N + 1)]
    for idx, triple in enumerate(bad_triples):
        for v in triple:
            triple_incidence[v].append(idx)
            
    # active_triple_count[v] = number of bad triples containing v that are currently active
    # active_triples = set of indices of bad triples that are currently active
    # Initially, all triples are active.
    
    def efficient_prune(random_seed=0):
        random.seed(random_seed)
        S_set = set(range(1, N + 1))
        active_triples = set(range(len(bad_triples)))
        
        # Count active triples per vertex
        active_count = [0] * (N + 1)
        for idx in active_triples:
            a, b, c = bad_triples[idx]
            active_count[a] += 1
            active_count[b] += 1
            active_count[c] += 1
            
        # We need to efficiently find a vertex with high active_count.
        # We can just iterate through vertices 1..N and pick the max.
        # N=128 is small.
        
        while True:
            if time.time() > DEADLINE - 2:
                break
                
            if not active_triples:
                break
                
            # Find vertex with max active_count
            max_count = 0
            victim = -1
            for v in range(1, N + 1):
                if v in S_set and active_count[v] > max_count:
                    max_count = active_count[v]
                    victim = v
                    
            if victim == -1:
                break
                
            # Remove victim
            S_set.discard(victim)
            
            # Update active_triples and active_count
            # For each triple containing victim, if it was active, it becomes inactive.
            # We need to know which triples containing victim are still active.
            # We can just iterate through triple_incidence[victim] and check if idx is in active_triples.
            for idx in triple_incidence[victim]:
                if idx in active_triples:
                    a, b, c = bad_triples[idx]
                    # This triple is now inactive
                    active_triples.discard(idx)
                    # Decrement counts for the other two vertices
                    # Note: we already decremented victim's count implicitly by removing it from S_set?
                    # No, active_count[v] is the number of ACTIVE triples containing v.
                    # When a triple becomes inactive, we must decrement counts for ALL 3 vertices in it.
                    active_count[a] -= 1
                    active_count[b] -= 1
                    active_count[c] -= 1
                    
        return S_set

    def local_search_add(S_set, random_seed=0):
        random.seed(random_seed)
        S_set = set(S_set)
        all_elements = set(range(1, N + 1))
        removed = all_elements - S_set
        candidates = list(removed)
        random.shuffle(candidates)
        
        for x in candidates:
            if time.time() > DEADLINE - 2:
                break
            can_add = True
            for idx in triple_incidence[x]:
                a, b, c = bad_triples[idx]
                if x == a:
                    others = (b, c)
                elif x == b:
                    others = (a, c)
                else:
                    others = (a, b)
                if others[0] in S_set and others[1] in S_set:
                    can_add = False
                    break
            if can_add:
                S_set.add(x)
        return S_set

    best_solution = set()
    
    # Run multiple restarts with different random seeds for tie-breaking or initial shuffling?
    # The efficient_prune is deterministic if we always pick the first max.
    # To diversify, we can break ties randomly.
    
    num_restarts = 50
    for i in range(num_restarts):
        if time.time() > DEADLINE - 5:
            break
            
        # Modify efficient_prune to accept a random seed for tie-breaking
        def efficient_prune_random(random_seed):
            random.seed(random_seed)
            S_set = set(range(1, N + 1))
            active_triples = set(range(len(bad_triples)))
            
            active_count = [0] * (N + 1)
            for idx in active_triples:
                a, b, c = bad_triples[idx]
                active_count[a] += 1
                active_count[b] += 1
                active_count[c] += 1
                
            while True:
                if time.time() > DEADLINE - 2:
                    break
                if not active_triples:
                    break
                    
                # Find vertices with max active_count
                max_count = 0
                for v in range(1, N + 1):
                    if v in S_set and active_count[v] > max_count:
                        max_count = active_count[v]
                        
                candidates = [v for v in range(1, N + 1) if v in S_set and active_count[v] == max_count]
                if not candidates:
                    break
                victim = random.choice(candidates)
                
                S_set.discard(victim)
                
                for idx in triple_incidence[victim]:
                    if idx in active_triples:
                        a, b, c = bad_triples[idx]
                        active_triples.discard(idx)
                        active_count[a] -= 1
                        active_count[b] -= 1
                        active_count[c] -= 1
                        
            return S_set

        S_set = efficient_prune_random(i)
        S_set = local_search_add(S_set, random_seed=i + 1000)
        
        if len(S_set) > len(best_solution):
            best_solution = S_set

    if not best_solution:
        best_solution = {1}
        
    return list(best_solution)

# Wrap to ensure nonempty
_orig_construct = construct
def construct():
    result = _orig_construct()
    if not result:
        return [1]
    return result
