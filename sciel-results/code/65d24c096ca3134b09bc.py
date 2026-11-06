import time
import random
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    # Moduli range
    M_MIN = 20
    M_MAX = 200
    moduli = list(range(M_MIN, M_MAX + 1))
    
    # State: dict mapping m -> set of residues
    # Or list of (m, r) for faster iteration?
    # Dict is better for "all residues for m" logic.
    
    def is_valid_add(m, r, state):
        for m_e, r_set in state.items():
            if m_e == m:
                # Same modulus: r must not be in r_set
                if r in r_set:
                    return False
            else:
                g = gcd(m, m_e)
                # Check if r conflicts with any r_e in r_set
                # r % g == r_e % g
                target = r % g
                for r_e in r_set:
                    if r_e % g == target:
                        return False
        return True

    def get_score(state):
        score = 0.0
        for m, r_set in state.items():
            score += len(r_set) / m
        return score

    # Greedy Initialization
    # Sort moduli by 1/m descending (m ascending)
    state = {}
    for m in sorted(moduli):
        # Try to add all residues
        # But adding one might block others?
        # For same m, residues are disjoint.
        # So we can add all residues that are valid against OTHER moduli.
        valid_residues = []
        for r in range(m):
            # Check against all existing state
            ok = True
            for m_e, r_set in state.items():
                g = gcd(m, m_e)
                target = r % g
                for r_e in r_set:
                    if r_e % g == target:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                valid_residues.append(r)
        
        if valid_residues:
            state[m] = set(valid_residues)
    
    best_state = {m: set(r_set) for m, r_set in state.items()}
    best_score = get_score(best_state)
    
    # Simulated Annealing
    T = 1.0
    T_min = 0.001
    cooling = 0.9999
    random.seed(42)
    
    current_state = {m: set(r_set) for m, r_set in state.items()}
    current_score = get_score(current_state)
    
    while T > T_min and time.time() < deadline:
        # Choose a move
        # 1. Add residue
        # 2. Remove residue
        # 3. Swap residue (remove one, add another for same m)
        # 4. Change modulus (remove all for m, add all valid for new m? No, too big)
        
        move_type = random.random()
        
        if move_type < 0.4:
            # Add residue
            if not current_state:
                continue
            m = random.choice(list(current_state.keys()))
            # Pick a residue not in current set
            possible = [r for r in range(m) if r not in current_state[m]]
            if not possible:
                continue
            r = random.choice(possible)
            if is_valid_add(m, r, current_state):
                new_state = {m_e: set(r_e) for m_e, r_e in current_state.items()}
                new_state[m].add(r)
                new_score = get_score(new_state)
                delta = new_score - current_score
                if delta > 0 or random.random() < pow(2.718281828, delta / T):
                    current_state = new_state
                    current_score = new_score
                    if current_score > best_score:
                        best_state = {m_e: set(r_e) for m_e, r_e in current_state.items()}
                        best_score = current_score
        elif move_type < 0.8:
            # Remove residue
            if not current_state:
                continue
            m = random.choice(list(current_state.keys()))
            if not current_state[m]:
                del current_state[m]
                continue
            r = random.choice(list(current_state[m]))
            new_state = {m_e: set(r_e) for m_e, r_e in current_state.items()}
            new_state[m].remove(r)
            if not new_state[m]:
                del new_state[m]
            new_score = get_score(new_state)
            delta = new_score - current_score
            if delta > 0 or random.random() < pow(2.718281828, delta / T):
                current_state = new_state
                current_score = new_score
                if current_score > best_score:
                    best_state = {m_e: set(r_e) for m_e, r_e in current_state.items()}
                    best_score = current_score
        else:
            # Add new modulus with one residue
            # Pick a modulus not in state, or in state?
            # If in state, it's "Add residue".
            # So pick a modulus NOT in state.
            used_mods = set(current_state.keys())
            available_mods = [m for m in moduli if m not in used_mods]
            if not available_mods:
                continue
            m = random.choice(available_mods)
            # Try to find a valid residue
            found = False
            for _ in range(10):
                r = random.randint(0, m - 1)
                if is_valid_add(m, r, current_state):
                    new_state = {m_e: set(r_e) for m_e, r_e in current_state.items()}
                    new_state[m] = {r}
                    new_score = get_score(new_state)
                    delta = new_score - current_score
                    if delta > 0 or random.random() < pow(2.718281828, delta / T):
                        current_state = new_state
                        current_score = new_score
                        if current_score > best_score:
                            best_state = {m_e: set(r_e) for m_e, r_e in current_state.items()}
                            best_score = current_score
                    found = True
                    break
            
        T *= cooling
    
    # Convert to list of [m, r]
    result = []
    for m, r_set in best_state.items():
        for r in r_set:
            result.append([m, r])
    
    return result
