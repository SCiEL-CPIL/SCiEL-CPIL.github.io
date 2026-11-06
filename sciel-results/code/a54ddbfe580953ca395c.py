import time
import math
import random
from math import gcd
from collections import deque

def construct():
    deadline = time.time() + 115  # Safety margin for 150s limit
    
    min_m = 20
    max_m = 200
    moduli = list(range(min_m, max_m + 1))
    n_moduli = len(moduli)
    mod_idx = {m: i for i, m in enumerate(moduli)}
    
    # Precompute GCDs
    gcd_table = {}
    for i, m1 in enumerate(moduli):
        for j, m2 in enumerate(moduli):
            gcd_table[(m1, m2)] = math.gcd(m1, m2)
    
    # Precompute 1/m as float
    inv_m = {m: 1.0 / m for m in moduli}
    
    def get_valid_residues(m, state):
        """Get all valid residues for modulus m given current state {m2: r2}"""
        valid_rs = []
        for r in range(m):
            ok = True
            for m2, r2 in state.items():
                if m2 == m:
                    continue
                g = gcd_table.get((m, m2), math.gcd(m, m2))
                if r % g == r2 % g:
                    ok = False
                    break
            if ok:
                valid_rs.append(r)
        return valid_rs
    
    def check_valid(m, r, state, exclude_m=None):
        """Check if (m, r) is valid given state"""
        for m2, r2 in state.items():
            if m2 == m:
                continue
            if exclude_m is not None and m2 == exclude_m:
                continue
            g = gcd_table.get((m, m2), math.gcd(m, m2))
            if r % g == r2 % g:
                return False
        return True
    
    def ac3_prune(domains, selected_moduli):
        """
        AC-3 arc consistency to prune incompatible residue values.
        domains: dict {m: set of valid residues}
        selected_moduli: list of moduli that are in the current state
        Returns pruned domains or None if inconsistency detected
        """
        # Initialize queue with all arcs between selected moduli
        queue = deque()
        for i, m1 in enumerate(selected_moduli):
            for m2 in selected_moduli[i+1:]:
                if m1 in domains and m2 in domains:
                    queue.append((m1, m2))
                    queue.append((m2, m1))
        
        while queue:
            m1, m2 = queue.popleft()
            if m1 not in domains or m2 not in domains:
                continue
            if not domains[m1] or not domains[m2]:
                return None
            
            g = gcd_table.get((m1, m2), math.gcd(m1, m2))
            
            # Check if every value in domains[m1] has a compatible value in domains[m2]
            new_domain_m1 = set()
            for r1 in domains[m1]:
                for r2 in domains[m2]:
                    if r1 % g != r2 % g:
                        new_domain_m1.add(r1)
                        break
            
            if len(new_domain_m1) < len(domains[m1]):
                domains[m1] = new_domain_m1
                if not domains[m1]:
                    return None
                # Add arcs pointing to m1
                for m3 in selected_moduli:
                    if m3 != m1 and m3 in domains:
                        queue.append((m3, m1))
        
        return domains
    
    def weighted_maxsat_search(selected_moduli, max_time=20):
        """
        Weighted Max-SAT style search with AC-3 pruning.
        Tries to find optimal residue assignments for selected moduli.
        """
        if not selected_moduli:
            return {}, 0.0
        
        start_time = time.time()
        
        # Initialize domains: all residues 0..m-1 for each selected modulus
        domains = {m: set(range(m)) for m in selected_moduli}
        
        # Apply AC-3
        pruned = ac3_prune(domains, selected_moduli)
        if pruned is None:
            # Inconsistent - try removing some moduli
            return {}, 0.0
        
        # Greedy assignment with backtracking
        best_assignment = {}
        best_score = 0.0
        
        def backtrack(idx, current_assignment, current_score):
            nonlocal best_assignment, best_score
            
            if time.time() - start_time > max_time:
                return
            
            if idx == len(selected_moduli):
                if current_score > best_score:
                    best_score = current_score
                    best_assignment = dict(current_assignment)
                return
            
            m = selected_moduli[idx]
            # Try residues in order of preference (smaller residues first for determinism)
            for r in sorted(domains.get(m, [])):
                # Check compatibility with already assigned moduli
                ok = True
                for m2, r2 in current_assignment.items():
                    g = gcd_table.get((m, m2), math.gcd(m, m2))
                    if r % g == r2 % g:
                        ok = False
                        break
                
                if ok:
                    current_assignment[m] = r
                    backtrack(idx + 1, current_assignment, current_score + inv_m[m])
                    del current_assignment[m]
        
        # Sort moduli by 1/m descending to prioritize high-value moduli
        sorted_moduli = sorted(selected_moduli, key=lambda m: -inv_m[m])
        backtrack(0, {}, 0.0)
        
        return best_assignment, best_score
    
    def local_search_with_pruning(initial_state, max_time=30):
        """
        Local search that uses AC-3 pruning to guide moves.
        """
        current_state = dict(initial_state)
        current_score = sum(inv_m[m] for m in current_state)
        best_state = dict(current_state)
        best_score = current_score
        
        start_time = time.time()
        temp = 0.02
        cooling_rate = 0.999
        
        while time.time() - start_time < max_time and temp > 0.0001:
            if time.time() > deadline - 5:
                break
            
            move_type = random.random()
            
            if move_type < 0.35:
                # Add a modulus
                unselected = [m for m in moduli if m not in current_state]
                if not unselected:
                    temp *= cooling_rate
                    continue
                
                # Weight by 1/m^2 to prefer small moduli
                weights = [inv_m[m] ** 2 for m in unselected]
                total_w = sum(weights)
                r_rand = random.random() * total_w
                cum = 0
                m_new = unselected[-1]
                for m, w in zip(unselected, weights):
                    cum += w
                    if r_rand <= cum:
                        m_new = m
                        break
                
                # Get valid residues using AC-3 on the subset
                candidate_moduli = list(current_state.keys()) + [m_new]
                domains = {m: set(range(m)) for m in candidate_moduli}
                
                # Set fixed residues for already-assigned moduli
                for m, r in current_state.items():
                    domains[m] = {r}
                
                pruned = ac3_prune(domains, candidate_moduli)
                if pruned is not None and pruned[m_new]:
                    valid_rs = list(pruned[m_new])
                    if valid_rs:
                        r_new = random.choice(valid_rs)
                        new_state = dict(current_state)
                        new_state[m_new] = r_new
                        new_score = current_score + inv_m[m_new]
                        delta = new_score - current_score
                        
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            current_state = new_state
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = dict(current_state)
                else:
                    # Fallback: simple check
                    valid_rs = get_valid_residues(m_new, current_state)
                    if valid_rs:
                        r_new = random.choice(valid_rs)
                        new_state = dict(current_state)
                        new_state[m_new] = r_new
                        new_score = current_score + inv_m[m_new]
                        delta = new_score - current_score
                        
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            current_state = new_state
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = dict(current_state)
                            
            elif move_type < 0.50:
                # Remove a modulus
                if not current_state:
                    temp *= cooling_rate
                    continue
                keys = list(current_state.keys())
                weights = [inv_m[m] for m in keys]
                total_w = sum(weights)
                r_rand = random.random() * total_w
                cum = 0
                m_old = keys[-1]
                for m, w in zip(keys, weights):
                    cum += w
                    if r_rand <= cum:
                        m_old = m
                        break
                
                new_state = dict(current_state)
                del new_state[m_old]
                new_score = current_score - inv_m[m_old]
                delta = new_score - current_score
                
                if delta > 0 or random.random() < math.exp(delta / temp):
                    current_state = new_state
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_state = dict(current_state)
                            
            elif move_type < 0.75:
                # Swap: remove one, add another with AC-3 validation
                if not current_state:
                    temp *= cooling_rate
                    continue
                keys = list(current_state.keys())
                weights_rem = [inv_m[m] for m in keys]
                total_w_rem = sum(weights_rem)
                r_rand_rem = random.random() * total_w_rem
                cum = 0
                m_old = keys[-1]
                for m, w in zip(keys, weights_rem):
                    cum += w
                    if r_rand_rem <= cum:
                        m_old = m
                        break
                
                unselected = [m for m in moduli if m not in current_state]
                if not unselected:
                    temp *= cooling_rate
                    continue
                weights_add = [inv_m[m] ** 2 for m in unselected]
                total_w_add = sum(weights_add)
                r_rand_add = random.random() * total_w_add
                cum = 0
                m_new = unselected[-1]
                for m, w in zip(unselected, weights_add):
                    cum += w
                    if r_rand_add <= cum:
                        m_new = m
                        break
                
                # Try swap with AC-3
                temp_state = dict(current_state)
                del temp_state[m_old]
                
                candidate_moduli = list(temp_state.keys()) + [m_new]
                domains = {m: set(range(m)) for m in candidate_moduli}
                for m, r in temp_state.items():
                    domains[m] = {r}
                
                pruned = ac3_prune(domains, candidate_moduli)
                if pruned is not None and pruned[m_new]:
                    valid_rs = list(pruned[m_new])
                    if valid_rs:
                        r_new = random.choice(valid_rs)
                        new_state = dict(temp_state)
                        new_state[m_new] = r_new
                        new_score = current_score - inv_m[m_old] + inv_m[m_new]
                        delta = new_score - current_score
                        
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            current_state = new_state
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = dict(current_state)
                else:
                    # Fallback
                    valid_rs = get_valid_residues(m_new, temp_state)
                    if valid_rs:
                        r_new = random.choice(valid_rs)
                        new_state = dict(temp_state)
                        new_state[m_new] = r_new
                        new_score = current_score - inv_m[m_old] + inv_m[m_new]
                        delta = new_score - current_score
                        
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            current_state = new_state
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = dict(current_state)
                            
            elif move_type < 0.90:
                # Change residue of a random selected modulus with AC-3 validation
                if not current_state:
                    temp *= cooling_rate
                    continue
                m_var = random.choice(list(current_state.keys()))
                old_r = current_state[m_var]
                
                # Get all potentially valid residues for m_var
                other_state = {k: v for k, v in current_state.items() if k != m_var}
                valid_rs = get_valid_residues(m_var, other_state)
                
                if valid_rs and len(valid_rs) > 1:
                    new_r = random.choice([r for r in valid_rs if r != old_r])
                    if check_valid(m_var, new_r, other_state):
                        new_state = dict(current_state)
                        new_state[m_var] = new_r
                        current_state = new_state
                        
            else:
                # Double swap with AC-3
                if len(current_state) < 2:
                    temp *= cooling_rate
                    continue
                keys = list(current_state.keys())
                m_old1 = random.choice(keys)
                remaining = [k for k in keys if k != m_old1]
                m_old2 = random.choice(remaining)
                
                unselected = [m for m in moduli if m not in current_state]
                if len(unselected) < 2:
                    temp *= cooling_rate
                    continue
                m_new1 = random.choice(unselected)
                remaining_unsel = [m for m in unselected if m != m_new1]
                m_new2 = random.choice(remaining_unsel)
                
                temp_state = dict(current_state)
                del temp_state[m_old1]
                del temp_state[m_old2]
                
                # Try adding m_new1 first
                valid_rs1 = get_valid_residues(m_new1, temp_state)
                if valid_rs1:
                    r_new1 = random.choice(valid_rs1)
                    temp_state[m_new1] = r_new1
                    
                    # Then try m_new2
                    valid_rs2 = get_valid_residues(m_new2, temp_state)
                    if valid_rs2:
                        r_new2 = random.choice(valid_rs2)
                        temp_state[m_new2] = r_new2
                        
                        new_score = current_score - inv_m[m_old1] - inv_m[m_old2] + inv_m[m_new1] + inv_m[m_new2]
                        delta = new_score - current_score
                        
                        if delta > 0 or random.random() < math.exp(delta / temp):
                            current_state = temp_state
                            current_score = new_score
                            if current_score > best_score:
                                best_score = current_score
                                best_state = dict(current_state)
            
            temp *= cooling_rate
        
        return best_state, best_score
    
    def greedy_construct_csp(order=None, randomize=False):
        """Greedy construction with CSP-based residue selection."""
        if order is None:
            order = moduli[:]
            if randomize:
                random.shuffle(order)
        
        state = {}
        for m in order:
            if time.time() > deadline - 30:
                break
            
            # Use AC-3 to find valid residues
            candidate_moduli = list(state.keys()) + [m]
            domains = {mm: set(range(mm)) for mm in candidate_moduli}
            for mm, rr in state.items():
                domains[mm] = {rr}
            
            pruned = ac3_prune(domains, candidate_moduli)
            
            if pruned is not None and pruned[m]:
                valid_rs = list(pruned[m])
                if valid_rs:
                    if randomize:
                        r = random.choice(valid_rs)
                    else:
                        r = valid_rs[0]
                    state[m] = r
            else:
                # Fallback to simple check
                valid_rs = get_valid_residues(m, state)
                if valid_rs:
                    if randomize:
                        r = random.choice(valid_rs)
                    else:
                        r = valid_rs[0]
                    state[m] = r
        return state
    
    best_solution = {}
    best_score = 0.0
    
    # Strategy 1: Greedy sorted by 1/m descending with CSP
    order_desc = sorted(moduli, key=lambda m: -inv_m[m])
    state1 = greedy_construct_csp(order=order_desc)
    sol1, score1 = local_search_with_pruning(state1, max_time=25)
    if score1 > best_score:
        best_score = score1
        best_solution = sol1
    
    # Strategy 2: Greedy sorted ascending with CSP
    order_asc = sorted(moduli, key=lambda m: inv_m[m])
    state2 = greedy_construct_csp(order=order_asc)
    sol2, score2 = local_search_with_pruning(state2, max_time=15)
    if score2 > best_score:
        best_score = score2
        best_solution = sol2
    
    # Strategy 3-6: Randomized greedy + local search with CSP
    num_restarts = 5
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        state = greedy_construct_csp(randomize=True)
        sol, score = local_search_with_pruning(state, max_time=12)
        if score > best_score:
            best_score = score
            best_solution = sol
    
    # Final polish on best solution
    if time.time() < deadline - 8 and best_solution:
        sol_final, score_final = local_search_with_pruning(best_solution, max_time=8)
        if score_final > best_score:
            best_score = score_final
            best_solution = sol_final
    
    if best_solution:
        return [[m, r] for m, r in best_solution.items()]
    
    # Fallback
    state = greedy_construct_csp(randomize=False)
    return [[m, r] for m, r in state.items()]

if __name__ == "__main__":
    start = time.time()
    res = construct()
    end = time.time()
    score = sum(1.0 / m for m, _ in res)
    print(f"Score: {score}")
    print(f"Time: {end - start:.2f}s")
    print(f"Classes: {len(res)}")
    valid = True
    for i in range(len(res)):
        for j in range(i+1, len(res)):
            m1, r1 = res[i]
            m2, r2 = res[j]
            g = gcd(m1, m2)
            if r1 % g == r2 % g:
                valid = False
                print(f"Conflict: ({m1},{r1}) and ({m2},{r2})")
    print(f"Valid: {valid}")
