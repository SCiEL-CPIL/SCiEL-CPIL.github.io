<<<<<<< SEARCH
import time
import random
import sys
from itertools import combinations, permutations
=======
import time
import random
import sys
from itertools import combinations, permutations
import numpy as np
>>>>>>> REPLACE
<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
=======
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5 second safety margin
    np.random.seed(42)
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Greedy Set Cover on the complement
    # We want to remove edges to hit all Q3s.
    active = [True] * num_q3s
    counts = [0] * num_edges_total
    
    # Build lists: q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    # Initialize counts
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    removed_edges = [False] * num_edges_total
    
    # Greedy Loop
    while True:
        if time.time() > deadline:
            break
            
        # Find edge with max count
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if removed_edges[e]:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        # Remove best_edge
        removed_edges[best_edge] = True
        
        # Update active Q3s and counts
        for q3_idx in q3s_containing_edge[best_edge]:
            if active[q3_idx]:
                active[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb
=======
    # Build lists: q3s_containing_edge[e] = list of indices of Q3s containing e
    q3s_containing_edge = [[] for _ in range(num_edges_total)]
    
    for i, mask in enumerate(global_q3_masks):
        m = mask
        while m:
            lsb = m & -m
            e_idx = lsb.bit_length() - 1
            q3s_containing_edge[e_idx].append(i)
            m ^= lsb
            
    # Convert global_q3_masks to numpy array for vectorized operations
    # Each mask is a 120-bit integer; we'll store as Python ints but use numpy for edge presence
    # Build a matrix: q3_edge_matrix[q3_idx, e_idx] = 1 if edge e_idx is in Q3 q3_idx
    # This is 10.8M * 120 bits = too large for a dense matrix.
    # Instead, use the list structure.
    
    removed_edges = [False] * num_edges_total
    removed_set = set()
    
    # Greedy Set Cover on the complement
    # We want to remove edges to hit all Q3s.
    active = [True] * num_q3s
    counts = [0] * num_edges_total
    
    # Initialize counts
    for e in range(num_edges_total):
        counts[e] = len(q3s_containing_edge[e])
        
    # Greedy Loop
    while True:
        if time.time() > deadline:
            break
            
        # Find edge with max count
        max_count = -1
        best_edge = -1
        for e in range(num_edges_total):
            if removed_edges[e]:
                continue
            if counts[e] > max_count:
                max_count = counts[e]
                best_edge = e
                
        if best_edge == -1 or max_count == 0:
            break
            
        # Remove best_edge
        removed_edges[best_edge] = True
        removed_set.add(best_edge)
        
        # Update active Q3s and counts
        for q3_idx in q3s_containing_edge[best_edge]:
            if active[q3_idx]:
                active[q3_idx] = False
                mask = global_q3_masks[q3_idx]
                m = mask
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != best_edge:
                        counts[e_idx] -= 1
                    m ^= lsb
>>>>>>> REPLACE
<<<<<<< SEARCH
    # Local search polish: try edge swaps to reduce remaining Q3 count
    # An edge swap: add back a removed edge 'a', remove a kept edge 'b'
    # This is beneficial if the number of Q3s newly created by adding 'a'
    # (among Q3s that still need 'a' to be hit) is less than the number of Q3s
    # that were only hit by 'b' (i.e., removing 'b' unhits them) minus overlap.
    
    # More precisely: after greedy, we have a set of removed edges R.
    # The graph G = K16 - R is Q3-free.
    # We want to try: for each a in R, b not in R:
    #   new_R = R - {a} + {b}
    #   This is valid if G' = K16 - new_R is Q3-free.
    #   G' has one more edge than G, so we gain 1 edge.
    #   G' is Q3-free iff no Q3 is entirely contained in G'.
    #   A Q3 is in G' iff all its 12 edges are in G'.
    #   Edges in G' = all edges except (R - {a} + {b}).
    #   So a Q3 is in G' iff it contains none of (R - {a} + {b}).
    #   Equivalently: it does NOT contain any edge from R\{a}, AND does NOT contain b.
    #   But it DOES contain a (since a is no longer removed).
    #   
    #   Let's think differently. After greedy, active[q3] is False for all q3
    #   (since greedy hits all). Actually no - greedy stops when max_count==0,
    #   meaning all remaining Q3s have count 0, which means they're all hit.
    #   Wait, let me re-examine. The greedy removes edges until every Q3 has at
    #   least one removed edge. So after greedy, for every Q3, at least one of
    #   its edges is in R.
    #   
    #   For a swap (add a, remove b):
    #   A Q3 becomes "unhit" (present in G') iff:
    #     - It contains a (so a being added back doesn't help hit it)
    #     - It does NOT contain any other edge from R\{a} (otherwise still hit)
    #     - It does NOT contain b (otherwise b hits it)
    #   Wait, that's not right either. Let me think more carefully.
    #   
    #   G' contains all edges except R' = (R \ {a}) ∪ {b}.
    #   A Q3 Q is present in G' iff every edge of Q is in G', i.e., no edge of Q is in R'.
    #   Q is present iff: 
    #     (Q ∩ (R \ {a})) = ∅  AND  b ∉ Q
    #   
    #   So the number of Q3s present in G' = |{Q : (Q ∩ (R\{a})) = ∅ and b ∉ Q}|
    #   
    #   This is expensive to compute naively. Let's use a different approach.
    #   
    #   Alternative: just verify validity by checking all Q3s after each swap.
    #   But that's 12870*840 = ~10.8M checks per swap, too slow.
    #   
    #   Better approach: precompute for each edge e, the set of Q3s that contain e.
    #   We already have q3s_containing_edge.
    #   
    #   For the current removed set R, a Q3 is hit iff it shares at least one edge with R.
    #   For swap (add a, remove b):
    #   Q3 Q is unhit in G' iff (Q ∩ R') = ∅ where R' = (R\{a}) ∪ {b}
    #   iff (Q ∩ (R\{a})) = ∅ AND (Q ∩ {b}) = ∅
    #   iff (Q ∩ R) ⊆ {a} AND b ∉ Q
    #   
    #   So Q is unhit iff: Q contains a, Q contains no other edge from R, and b is not in Q.
    #   Wait: (Q ∩ R) ⊆ {a} means Q∩R is either ∅ or {a}.
    #   If Q∩R = ∅: then Q was already unhit before the swap, which shouldn't happen
    #   since greedy hit all Q3s. So Q∩R must be non-empty.
    #   Actually after greedy, every Q3 has Q∩R ≠ ∅.
    #   So (Q ∩ R) ⊆ {a} and Q∩R ≠ ∅ means Q∩R = {a}.
    #   
    #   So: Q is unhit in G' iff (Q ∩ R) = {a} AND b ∉ Q.
    #   
    #   This is much more tractable! For each a in R, the Q3s that are "critically hit only by a"
    #   are those with Q∩R = {a}. Let's call this set C_a.
    #   Then for swap (a, b): number of unhit Q3s = |{Q in C_a : b ∉ Q}|
    #   The swap is valid iff this count is 0, i.e., every Q in C_a contains b.
    #   
    #   We can precompute C_a for all a in R.
    
    if time.time() < deadline - 5:
        # Precompute C_a: for each removed edge a, the list of Q3s where Q∩R = {a}
        # For a Q3 to have Q∩R = {a}, it must contain a and no other edge from R.
        
        # First, for each Q3, compute which removed edges it contains.
        # q3_removed_edges[q3_idx] = list of removed edges in this Q3
        # But we need this efficiently.
        
        # Alternative: for each a in R, C_a = {Q : a ∈ Q and (Q ∩ R) = {a}}
        # = {Q : a ∈ Q} \ {Q : a ∈ Q and ∃ e' ∈ R\{a} with e' ∈ Q}
        
        # Let's compute for each a in R:
        #   all_q3s_with_a = q3s_containing_edge[a]
        #   For each Q in all_q3s_with_a, check if Q has any other edge in R.
        
        # This is expensive if done naively. Let's limit to a few iterations.
        
        # Simpler approach: just try random swaps and verify by rechecking.
        # Or: use the counts array. After greedy, counts[e] for e not in R 
        # gives the number of Q3s that contain e and are still "active" in some sense.
        # Actually counts[e] was updated during greedy. Let me re-examine.
        
        # After greedy, for each edge e not in R:
        #   counts[e] = number of Q3s Q such that e ∈ Q and Q was "hit" by some edge 
        #   that was removed after e's count was last updated. This is messy.
        
        # Let's just do a simple local search: try adding back each removed edge 
        # and check if the graph is still Q3-free by verifying all Q3s.
        # To make this fast, we'll only try a limited number of swaps.
        
        # Actually, let's try a different polish: for each removed edge a,
        # try to find if we can "swap" it with some kept edge b such that
        # the resulting graph is still Q3-free.
        
        # For efficiency, let's precompute for each Q3 its edge list as a frozenset.
        # We already have global_q3_masks. Let's convert to frozensets of edge indices.
        
        # This might be memory-heavy: 12870*840 = 10.8M Q3s, each with 12 edges.
        # 10.8M * 12 * 8 bytes ≈ 1GB. Too much.
        
        # Let's skip the heavy precomputation and do a lighter polish.
        # Try: for each removed edge a, check if removing a from R (i.e., adding it back)
        # keeps the graph Q3-free. If so, we've found an improvement.
        # But we need to also remove some other edge to keep the count the same... 
        # Actually we want to ADD an edge, so we don't need to remove one.
        # Wait, the greedy removed edges to make the graph Q3-free. 
        # If we can add back a removed edge and still be Q3-free, that's an improvement.
        
        # Check: is K16 - (R \ {a}) Q3-free?
        # This is Q3-free iff no Q3 has all its edges in K16 - (R\{a}).
        # I.e., no Q3 is disjoint from R\{a}.
        # I.e., every Q3 contains at least one edge from R\{a}.
        
        # For each a in R: check if every Q3 contains some edge from R\{a}.
        # Equivalent: no Q3 is "only hit by a" (i.e., Q∩R = {a}).
        
        # So we just need to check if C_a is empty for some a.
        
        # Let's compute C_a efficiently.
        # For each Q3, find which removed edges it contains.
        # If it contains exactly one removed edge a, then Q3 ∈ C_a.
        
        # Build: for each Q3, count how many removed edges it contains, and which one if 1.
        # This requires iterating over all Q3s and their 12 edges.
        
        # 10.8M Q3s * 12 edges = 130M operations. Might be too slow.
        
        # Let's limit: only consider Q3s that contain at least one removed edge.
        # Actually all Q3s contain at least one removed edge (greedy hit all).
        
        # Alternative: iterate over removed edges a, and for each, iterate over Q3s containing a,
        # and check if any other removed edge is also in that Q3.
        
        # Total work: sum over a in R of |q3s_containing_edge[a]| * (check for other removed edges)
        # |q3s_containing_edge[a]| is at most ~10.8M / 120 ≈ 90K per edge.
        # For each such Q3, checking if it has another removed edge requires scanning its 12 edges.
        # Total: |R| * 90K * 12 ≈ 20 * 90K * 12 ≈ 21.6M operations. Feasible.
        
        # Let's do this.
        
        # First, build a set of removed edges for fast lookup.
        removed_set = set()
        for e in range(num_edges_total):
            if removed_edges[e]:
                removed_set.add(e)
        
        # For each removed edge a, determine if C_a is empty.
        # C_a is empty iff every Q3 containing a also contains some other removed edge.
        
        improvable_edges = []
        for a in removed_set:
            only_hit_by_a = False
            for q3_idx in q3s_containing_edge[a]:
                # Check if this Q3 contains any other removed edge besides a
                mask = global_q3_masks[q3_idx]
                m = mask
                has_other_removed = False
                while m:
                    lsb = m & -m
                    e_idx = lsb.bit_length() - 1
                    if e_idx != a and e_idx in removed_set:
                        has_other_removed = True
                        break
                    m ^= lsb
                if not has_other_removed:
                    only_hit_by_a = True
                    break
            if not only_hit_by_a:
                improvable_edges.append(a)
        
        # For each improvable edge, add it back (remove from removed_set)
        # This increases edge count by 1 each.
        for a in improvable_edges:
            removed_edges[a] = False
            removed_set.discard(a)
        
        # Repeat polish for a few iterations
        polish_iter = 0
        while polish_iter < 3 and time.time() < deadline - 2:
            polish_iter += 1
            found_improvement = False
            for a in list(removed_set):
                only_hit_by_a = False
                for q3_idx in q3s_containing_edge[a]:
                    mask = global_q3_masks[q3_idx]
                    m = mask
                    has_other_removed = False
                    while m:
                        lsb = m & -m
                        e_idx = lsb.bit_length() - 1
                        if e_idx != a and e_idx in removed_set:
                            has_other_removed = True
                            break
                        m ^= lsb
                    if not has_other_removed:
                        only_hit_by_a = True
                        break
                if not only_hit_by_a:
                    removed_edges[a] = False
                    removed_set.discard(a)
                    found_improvement = True
            if not found_improvement:
                break
=======
    # Genetic Algorithm Polish
    # Start from the greedy solution and try to improve via GA
    
    def count_q3s_in_removed_set(removed_set):
        """Count Q3s that are NOT hit by the removed set (i.e., present in the graph)."""
        unhit_count = 0
        for q3_idx in range(num_q3s):
            mask = global_q3_masks[q3_idx]
            m = mask
            hit = False
            while m:
                lsb = m & -m
                e_idx = lsb.bit_length() - 1
                if e_idx in removed_set:
                    hit = True
                    break
                m ^= lsb
            if not hit:
                unhit_count += 1
        return unhit_count
    
    def fitness(removed_set):
        """Fitness = number of kept edges - large penalty for each unhit Q3."""
        num_removed = len(removed_set)
        num_kept = num_edges_total - num_removed
        unhit = count_q3s_in_removed_set(removed_set)
        return num_kept - 1000 * unhit
    
    # Current best solution from greedy
    current_removed = set(removed_set)
    current_fitness = fitness(current_removed)
    
    # GA parameters
    pop_size = 50
    mutation_rate = 0.1
    elite_count = 5
    
    # Initialize population with current solution and its perturbations
    population = [set(current_removed)]
    for _ in range(pop_size - 1):
        pop = set(current_removed)
        # Randomly flip some edges
        all_edges_list = list(range(num_edges_total))
        num_flips = random.randint(1, 5)
        for _ in range(num_flips):
            e = random.choice(all_edges_list)
            if e in pop:
                pop.discard(e)
            else:
                pop.add(e)
        population.append(pop)
    
    best_removed = set(current_removed)
    best_fitness = current_fitness
    
    gen = 0
    max_gens = 50
    
    while gen < max_gens and time.time() < deadline - 5:
        gen += 1
        
        # Evaluate fitness for all individuals
        fitnesses = [fitness(ind) for ind in population]
        
        # Update best
        for i, f in enumerate(fitnesses):
            if f > best_fitness:
                best_fitness = f
                best_removed = set(population[i])
        
        # Selection: keep elites, rest by tournament
        sorted_indices = sorted(range(pop_size), key=lambda i: fitnesses[i], reverse=True)
        
        new_population = [set(population[sorted_indices[i]]) for i in range(min(elite_count, pop_size))]
        
        while len(new_population) < pop_size:
            # Tournament selection
            a = random.randint(0, pop_size - 1)
            b = random.randint(0, pop_size - 1)
            if fitnesses[a] >= fitnesses[b]:
                parent1 = population[a]
            else:
                parent1 = population[b]
            
            a = random.randint(0, pop_size - 1)
            b = random.randint(0, pop_size - 1)
            if fitnesses[a] >= fitnesses[b]:
                parent2 = population[a]
            else:
                parent2 = population[b]
            
            # Crossover: uniform crossover on the edge set
            child = set()
            for e in range(num_edges_total):
                if e in parent1:
                    child.add(e)
                elif e in parent2:
                    if random.random() < 0.5:
                        child.add(e)
                else:
                    if random.random() < 0.5:
                        child.add(e)
            
            # Mutation
            for e in range(num_edges_total):
                if random.random() < mutation_rate:
                    if e in child:
                        child.discard(e)
                    else:
                        child.add(e)
            
            new_population.append(child)
        
        population = new_population
    
    # Use the best solution found
    final_removed = best_removed
    for e in range(num_edges_total):
        removed_edges[e] = e in final_removed
    
    # Final polish: try to add back edges that are only hit by themselves
    if time.time() < deadline - 2:
        removed_set_final = set(final_removed)
        polish_iter = 0
        while polish_iter < 5 and time.time() < deadline - 2:
            polish_iter += 1
            found_improvement = False
            for a in list(removed_set_final):
                only_hit_by_a = False
                for q3_idx in q3s_containing_edge[a]:
                    mask = global_q3_masks[q3_idx]
                    m = mask
                    has_other_removed = False
                    while m:
                        lsb = m & -m
                        e_idx = lsb.bit_length() - 1
                        if e_idx != a and e_idx in removed_set_final:
                            has_other_removed = True
                            break
                        m ^= lsb
                    if not has_other_removed:
                        only_hit_by_a = True
                        break
                if not only_hit_by_a:
                    removed_edges[a] = False
                    removed_set_final.discard(a)
                    found_improvement = True
            if not found_improvement:
                break
>>>>>>> REPLACE
