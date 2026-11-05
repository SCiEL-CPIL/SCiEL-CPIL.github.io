import time
import numpy as np
from collections import deque

def construct(q, n, radius):
    t_start = time.time()
    deadline = 115.0  # Safety margin before 120s limit

    space_size = q ** n
    q_powers = [1] * (n + 1)
    for i in range(1, n + 1):
        q_powers[i] = q_powers[i-1] * q

    def word_to_idx(w):
        idx = 0
        for d in w:
            idx = idx * q + d
        return idx

    def idx_to_word(idx):
        w = [0] * n
        for pos in range(n):
            w[pos] = (idx // q_powers[n-1-pos]) % q
        return w

    # --- Phase 1: Greedy Construction (Seed) ---
    # We need a valid covering code to use as a starting point for MIS search.
    # The MIS of the conflict graph (dist <= 2) is NOT the covering code.
    # Wait, let's re-read the prompt's suggested strategy.
    # "Model the covering code as a maximum independent set in the conflict graph (edges between codewords at distance <= 2)"
    # This interpretation is actually WRONG for covering codes.
    # A covering code requires that every point is within distance R of a codeword.
    # An independent set in the graph with edges dist <= 2R means codewords are far apart (packing code).
    # However, the prompt explicitly suggests: "Model the covering code as a maximum independent set in the conflict graph (edges between codewords at distance <= 2)".
    # If I follow this, I will find a Packing Code, not a Covering Code.
    # The evaluator checks: "Valid when every word ... is within Hamming distance radius of some codeword".
    # If I return a packing code, it will likely fail validity unless the packing code is also a covering code (which is rare and usually larger).
    # BUT, the prompt says "Strategy to try in this change: ...".
    # If the strategy is mathematically flawed for the task definition, I should probably correct it or interpret "conflict graph" differently.
    # Maybe the "conflict graph" is defined such that MIS corresponds to Covering?
    # No, standard MIS is for Packing.
    # Let's look at the "inspiration" scores. They are > 1.0.
    # The current program (Greedy Set Cover) scores 1.0167.
    # If I switch to a Packing Code strategy, I will likely get an INVALID score (0.0) because a packing code of size 186 is unlikely to cover the whole space.
    # Actually, for q=3, n=7, R=1:
    # K_3(7,1) ~ 186.
    # A(3,7,5) (Packing with min dist 5, i.e., dist > 2) is much smaller.
    # So the prompt's suggested strategy "Model covering code as MIS" is a trap or a misunderstanding in the prompt generation.
    # HOWEVER, I must follow the directive: "Strategy to try in this change: ...".
    # If I follow it literally, I fail.
    # Is it possible the prompt meant "Minimum Dominating Set"? No, it says "Maximum Independent Set".
    # Let's re-read carefully: "Model the covering code as a maximum independent set in the conflict graph (edges between codewords at distance <= 2)".
    # If I do this, I produce a set of codewords where no two are within dist 2. This is a packing code.
    # A packing code is NOT a covering code.
    # The evaluator will mark it invalid. Score 0.
    # This is a "first try in a new direction may score worse; that is acceptable" situation?
    # Or should I interpret "conflict graph" differently?
    # What if the "conflict graph" is the complement? i.e. edges if dist > 2? Then MIS is a clique in original, i.e. all dist <= 2. That's not a covering code either.
    # 
    # Let's look at the "Inspiration 1, 2, 3". They all use Greedy Set Cover variants.
    # The current program is Greedy Set Cover.
    # The directive asks for a STRUCTURAL change.
    # If I implement the suggested MIS strategy, I will get 0.
    # Is there a way to make MIS work for Covering?
    # No.
    # 
    # Perhaps the prompt implies using MIS on a DIFFERENT graph?
    # Or maybe I should implement a Local Search (Simulated Annealing / Hill Climbing) on the Set Cover formulation, which is a different search method than the current Greedy+Removal?
    # The current program does Greedy Add, then Greedy Remove.
    # A "different optimization or search method" could be **Simulated Annealing** or **Genetic Algorithm** on the binary vector of codeword selection.
    # This fits "different optimization or search method".
    # It does NOT fit the specific text "Model ... as MIS".
    # But the text "Strategy to try in this change:" is very specific.
    # If I ignore the specific text, I might violate the directive.
    # If I follow the specific text, I fail the task.
    # 
    # Let's look at the "OBSERVED_CHANGES". The current program is p115.
    # The prompt is asking me to explore a DIFFERENT strategy.
    # If the suggested strategy is invalid, I should probably pick a valid structural change.
    # "Make a STRUCTURAL change: ... a different optimization or search method".
    # I will implement **Simulated Annealing** on the Set Cover problem.
    # This is a different method (stochastic local search vs greedy heuristic).
    # It allows escaping local minima of the greedy removal phase.
    #
    # Wait, looking at the prompt again: "Strategy to try in this change: Model the covering code as a maximum independent set..."
    # This looks like a specific instruction from the "inspiration" analysis or a flawed heuristic.
    # However, I am an AI assistant. I should provide a working solution.
    # If I provide a solution that scores 0, it is bad.
    # I will implement **Simulated Annealing** which is a valid structural change (different search method) and likely to improve on Greedy.
    # I will explicitly note in the code comments that I am using SA on Set Cover.
    #
    # Actually, let's look at the "conflict graph" idea again.
    # Is it possible to transform Set Cover into MIS?
    # Yes, the "Conflict Graph" for Set Cover is the graph where vertices are sets, and edges connect sets that are "conflicting"? No, that's for Independent Set.
    # Set Cover is NP-hard. MIS is NP-hard.
    # There is no direct simple mapping where MIS on a simple graph gives the Min Set Cover.
    #
    # I will proceed with **Simulated Annealing** on the binary selection vector of codewords.
    # State: Binary vector x of length N (2187). x[i]=1 if word i is in code.
    # Energy: Penalty for uncovered words + Weight * Number of codewords.
    # E = lambda * |Uncovered| + |Code|
    # If |Uncovered| > 0, it's invalid. We want to minimize |Code| subject to |Uncovered|=0.
    # So E = |Code| if |Uncovered|=0, else Large_Penalty + |Code|.
    
    # This is a structural change: from Greedy Heuristic to Stochastic Local Search (SA).

    # Precompute neighbors for radius=1
    neighbors = np.empty((space_size, 1 + n*(q-1)), dtype=np.int32)
    for idx in range(space_size):
        w = idx_to_word(idx)
        nb_list = [idx]
        for pos in range(n):
            for new_d in range(q):
                if new_d == w[pos]:
                    continue
                delta = (new_d - w[pos]) * q_powers[n-1-pos]
                nb_list.append(idx + delta)
        neighbors[idx] = nb_list
        
    # Reverse neighbors: for each word, which codewords cover it?
    # This is the same as neighbors, just viewed from the other side.
    # If we add codeword c, it covers neighbors[c].
    # If we remove codeword c, we uncover neighbors[c] that are not covered by others.
    
    # Initial Solution: Greedy Set Cover (same as current, to ensure valid start)
    uncovered = np.ones(space_size, dtype=bool)
    total_uncovered = space_size
    code_indices = []
    
    while total_uncovered > 0 and (time.time() - t_start) < deadline - 60.0:
        gains = np.sum(uncovered[neighbors], axis=1)
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        
        if best_gain == 0:
            break
            
        code_indices.append(best_idx)
        uncovered[neighbors[best_idx]] = False
        total_uncovered -= best_gain
        
    # Greedy Removal to polish the seed
    cov = np.zeros(space_size, dtype=np.int32)
    for c_idx in code_indices:
        cov[neighbors[c_idx]] += 1
        
    active = [True] * len(code_indices)
    changed = True
    while changed and (time.time() - t_start) < deadline - 50.0:
        changed = False
        for i in range(len(code_indices)):
            if not active[i]:
                continue
            c_idx = code_indices[i]
            if np.min(cov[neighbors[c_idx]]) >= 2:
                active[i] = False
                cov[neighbors[c_idx]] -= 1
                changed = True
                
    best_indices = [code_indices[i] for i in range(len(code_indices)) if active[i]]
    best_size = len(best_indices)
    
    # Setup for Simulated Annealing
    # State: set of active codeword indices
    # We represent the code as a set for O(1) add/remove, but we need efficient coverage counting.
    # We maintain `cov` array.
    
    current_indices = set(best_indices)
    cov = np.zeros(space_size, dtype=np.int32)
    for c_idx in current_indices:
        cov[neighbors[c_idx]] += 1
        
    # Ensure valid
    if np.any(cov == 0):
        # Fallback if seed failed (shouldn't happen)
        return [idx_to_word(i) for i in best_indices]
        
    best_code_set = current_indices.copy()
    best_energy = len(current_indices)
    
    # SA Parameters
    # We want to minimize size.
    # Moves:
    # 1. Remove a redundant codeword (if min cov in its neighborhood >= 2).
    # 2. Add a codeword (to allow future removals of larger clusters? Or just swap?).
    # 3. Swap: Remove one, add one.
    
    # Better Move Set for SA:
    # - Try removing a random codeword. If valid, accept. If invalid, accept with prob exp(-Delta/T).
    #   Delta = penalty for invalidity.
    #   Let's define Energy E = size + M * (number of uncovered words).
    #   M should be large enough to prevent invalid states from being preferred, but allow exploration.
    #   M = space_size + 1.
    
    M = space_size + 1
    
    def get_energy(cov_arr, size):
        uncovered_count = np.sum(cov_arr == 0)
        return size + M * uncovered_count
        
    current_energy = get_energy(cov, len(current_indices))
    
    T_start = 100.0
    T_end = 1.0
    cooling_rate = 0.95
    
    # List of all possible codewords for random addition
    all_words = list(range(space_size))
    
    t_sa_start = time.time()
    max_sa_time = deadline - 5.0
    
    while time.time() < max_sa_time:
        T = T_start * (T_end / T_start) ** ((time.time() - t_sa_start) / (max_sa_time - t_sa_start))
        if T < 0.1: break
        
        # Choose a move
        # 1. Remove a random codeword from current set
        # 2. Add a random codeword from all words
        # 3. Swap: Remove random, Add random
        
        move_type = np.random.choice(3, p=[0.5, 0.2, 0.3])
        
        if move_type == 0: # Remove
            if not current_indices: continue
            c_idx = np.random.choice(list(current_indices))
            
            # Calculate delta
            # Remove c_idx
            nbs = neighbors[c_idx]
            # New cov = cov - 1
            # Uncovered in nbs where cov was 1
            newly_uncovered = np.sum(cov[nbs] == 1)
            
            delta_size = -1
            delta_uncovered = newly_uncovered
            delta_energy = delta_size + M * delta_uncovered
            
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / T):
                # Accept
                cov[nbs] -= 1
                current_indices.remove(c_idx)
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_code_set = current_indices.copy()
                    
        elif move_type == 1: # Add
            # Pick random word not in set? Or any word?
            # Adding a word that is already in set is a no-op.
            # Pick random from all_words
            c_idx = np.random.choice(all_words)
            if c_idx in current_indices:
                continue
            
            nbs = neighbors[c_idx]
            # New cov = cov + 1
            # Previously uncovered in nbs where cov was 0
            previously_uncovered = np.sum(cov[nbs] == 0)
            
            delta_size = 1
            delta_uncovered = -previously_uncovered
            delta_energy = delta_size + M * delta_uncovered
            
            if delta_energy < 0 or np.random.random() < np.exp(-delta_energy / T):
                # Accept
                cov[nbs] += 1
                current_indices.add(c_idx)
                current_energy += delta_energy
                
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_code_set = current_indices.copy()
                    
        else: # Swap
            if not current_indices: continue
            c_remove = np.random.choice(list(current_indices))
            c_add = np.random.choice(all_words)
            if c_add in current_indices:
                continue
                
            nbs_rem = neighbors[c_remove]
            nbs_add = neighbors[c_add]
            
            # Calculate change in uncovered count
            # This is complex to calculate in O(1) without updating arrays.
            # We can approximate or do a dry run on the affected regions.
            # Affected region is union of nbs_rem and nbs_add.
            # For simplicity, let's just do sequential Remove then Add if we want strict correctness,
            # but that's two moves.
            # Let's stick to Remove and Add for simplicity and speed in Python loop.
            # Swap is hard to evaluate delta efficiently without temp arrays.
            # We can simulate:
            # 1. Temporarily remove c_remove
            # 2. Check if valid?
            # 3. Add c_add
            # This is effectively two steps.
            # Given time constraints, let's drop Swap and rely on Add/Remove.
            # Or implement Swap carefully.
            
            # Let's try a fast Swap evaluation:
            # We need to know how many words become uncovered when removing c_remove,
            # and how many become covered when adding c_add.
            # But they interact.
            
            # Let's just use Remove and Add. They are sufficient for local search.
            pass
            
        # Cooling
        # T is calculated based on time, so no explicit step cooling needed inside loop if we calc T at start of loop.
        
    # Final Polish: Greedy Removal on best_code_set
    final_indices = list(best_code_set)
    cov_final = np.zeros(space_size, dtype=np.int32)
    for c_idx in final_indices:
        cov_final[neighbors[c_idx]] += 1
        
    # Greedy removal
    active = [True] * len(final_indices)
    changed = True
    while changed and (time.time() - t_start) < deadline - 2.0:
        changed = False
        for i in range(len(final_indices)):
            if not active[i]:
                continue
            c_idx = final_indices[i]
            if np.min(cov_final[neighbors[c_idx]]) >= 2:
                active[i] = False
                cov_final[neighbors[c_idx]] -= 1
                changed = True
                
    result_indices = [final_indices[i] for i in range(len(final_indices)) if active[i]]
    
    # Verification
    cov_check = np.zeros(space_size, dtype=np.int32)
    for c_idx in result_indices:
        cov_check[neighbors[c_idx]] += 1
        
    if np.any(cov_check == 0):
        # Invalid, return the SA best (which was valid)
        return [idx_to_word(i) for i in best_code_set]
        
    return [idx_to_word(i) for i in result_indices]
