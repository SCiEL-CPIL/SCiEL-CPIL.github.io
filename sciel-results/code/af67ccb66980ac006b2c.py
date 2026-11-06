"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import math
from math import gcd

def construct():
    """
    Maximizes sum(1/m) for pairwise disjoint residue classes with moduli in [20, 200].
    
    Strategy:
    1. Greedy initial construction (smallest moduli first)
    2. Residue reoptimization to free capacity
    3. Proper simulated annealing with temperature schedule
    4. Hill climbing polish
    """
    deadline = time.time() + 140.0
    random.seed(42)
    
    def _time_left():
        return deadline - time.time()
    
    def find_valid_residue(m, classes):
        """Find a residue r in [0, m) such that [m, r] is disjoint from all classes."""
        for r in range(m):
            valid = True
            for m2, r2 in classes:
                g = gcd(m, m2)
                if (r % g) == (r2 % g):
                    valid = False
                    break
            if valid:
                return r
        return None
    
    def is_valid_pair(m1, r1, m2, r2):
        """Check if two residue classes are disjoint."""
        g = gcd(m1, m2)
        return (r1 % g) != (r2 % g)
    
    def is_valid_config(classes):
        """Check if all pairs in classes are disjoint."""
        n = len(classes)
        for i in range(n):
            for j in range(i+1, n):
                if not is_valid_pair(classes[i][0], classes[i][1], classes[j][0], classes[j][1]):
                    return False
        return True
    
    def calc_score(classes):
        """Calculate the sum of 1/m as a float."""
        return sum(1.0 / m for m, r in classes)
    
    def try_add(classes, m):
        """Try to add modulus m with a valid residue. Returns new classes or None."""
        r = find_valid_residue(m, classes)
        if r is not None:
            return classes + [[m, r]]
        return None
    
    def residue_reoptimize(classes, time_limit=2.0):
        """Try to reassign residues to free up capacity for more moduli."""
        start = time.time()
        improved = True
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in range(20, 201) if m not in current_moduli]
            
            # Try to add missing moduli by reassigning residues
            for m_add in missing:
                if time.time() - start > time_limit:
                    break
                # Try each existing class and see if we can reassign its residue
                for i in range(len(classes)):
                    m_i, r_i = classes[i]
                    # Try all residues for class i
                    for r_new in range(m_i):
                        if r_new == r_i:
                            continue
                        # Check if new assignment is valid
                        valid = True
                        for j in range(len(classes)):
                            if j == i:
                                continue
                            if not is_valid_pair(m_i, r_new, classes[j][0], classes[j][1]):
                                valid = False
                                break
                        if not valid:
                            continue
                        # Try to add m_add with the new configuration
                        new_classes = classes[:]
                        new_classes[i] = [m_i, r_new]
                        r_add = find_valid_residue(m_add, new_classes)
                        if r_add is not None:
                            new_classes.append([m_add, r_add])
                            classes = new_classes
                            improved = True
                            break
                    if improved:
                        break
        return classes
    
    def hill_climb(classes, time_limit=3.0):
        """Hill climbing polish."""
        improved = True
        start = time.time()
        while improved and time.time() - start < time_limit:
            improved = False
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in range(20, 201) if m not in current_moduli]
            
            # Try adding
            for m in missing:
                if time.time() - start > time_limit:
                    break
                new_classes = try_add(classes, m)
                if new_classes is not None:
                    classes = new_classes
                    improved = True
            
            # Try swapping
            current_moduli = set(c[0] for c in classes)
            missing = [m for m in range(20, 201) if m not in current_moduli]
            for m_add in missing:
                if time.time() - start > time_limit:
                    break
                for m_remove in current_moduli:
                    if 1.0/m_add <= 1.0/m_remove:
                        continue
                    new_classes = [c for c in classes if c[0] != m_remove]
                    r = find_valid_residue(m_add, new_classes)
                    if r is not None:
                        new_classes.append([m_add, r])
                        classes = new_classes
                        improved = True
                        break
        return classes
    
    def sa_polish(classes, initial_temp=1.0, cooling=0.999, min_temp=0.01):
        """Proper simulated annealing with temperature schedule."""
        current = classes[:]
        current_score = calc_score(current)
        best = classes[:]
        best_score = current_score
        temperature = initial_temp
        
        while temperature > min_temp and _time_left() > 15.0:
            # Generate a random move
            move_type = random.random()
            delta = 0.0
            new_classes = None
            
            if move_type < 0.25:
                # Single residue reassignment
                if len(current) == 0:
                    temperature *= cooling
                    continue
                i = random.randint(0, len(current) - 1)
                m, r = current[i]
                r_new = random.randint(0, m - 1)
                if r_new == r:
                    temperature *= cooling
                    continue
                new_classes = current[:]
                new_classes[i] = [m, r_new]
                # Check validity
                valid = True
                for j in range(len(new_classes)):
                    if j == i:
                        continue
                    if not is_valid_pair(m, r_new, new_classes[j][0], new_classes[j][1]):
                        valid = False
                        break
                if not valid:
                    temperature *= cooling
                    continue
                delta = 0.0
            elif move_type < 0.5:
                # Add a missing modulus
                current_moduli = set(c[0] for c in current)
                missing = [m for m in range(20, 201) if m not in current_moduli]
                if not missing:
                    temperature *= cooling
                    continue
                m_add = random.choice(missing)
                r = find_valid_residue(m_add, current)
                if r is None:
                    temperature *= cooling
                    continue
                new_classes = current + [[m_add, r]]
                delta = 1.0 / m_add
            elif move_type < 0.7:
                # Remove a modulus
                if len(current) <= 2:
                    temperature *= cooling
                    continue
                i = random.randint(0, len(current) - 1)
                m_remove = current[i][0]
                new_classes = [c for c in current if c[0] != m_remove]
                delta = -1.0 / m_remove
            else:
                # 2-swap: swap one modulus for another
                current_moduli = set(c[0] for c in current)
                missing = [m for m in range(20, 201) if m not in current_moduli]
                if not missing or len(current) == 0:
                    temperature *= cooling
                    continue
                m_add = random.choice(missing)
                m_remove = random.choice([c[0] for c in current])
                new_classes = [c for c in current if c[0] != m_remove]
                r = find_valid_residue(m_add, new_classes)
                if r is None:
                    temperature *= cooling
                    continue
                new_classes.append([m_add, r])
                delta = 1.0/m_add - 1.0/m_remove
            
            if new_classes is None:
                temperature *= cooling
                continue
            
            new_score = current_score + delta
            
            # Acceptance criterion
            if new_score > current_score:
                current = new_classes
                current_score = new_score
                if current_score > best_score:
                    best = current[:]
                    best_score = current_score
            else:
                # Accept with probability exp(delta / temperature)
                if delta < 0:
                    accept_prob = math.exp(delta / temperature)
                else:
                    accept_prob = 1.0
                if random.random() < accept_prob:
                    current = new_classes
                    current_score = new_score
                    if current_score > best_score:
                        best = current[:]
                        best_score = current_score
            
            # Cool down
            temperature *= cooling
        
        return best, best_score
    
    # Initial greedy construction - try smallest moduli first
    classes = []
    for m in range(20, 201):
        if _time_left() < 10.0:
            break
        new_classes = try_add(classes, m)
        if new_classes is not None:
            classes = new_classes
    
    best_classes = classes[:]
    best_score = calc_score(classes)
    
    # Residue reoptimization to free capacity
    if _time_left() > 20.0:
        reopt = residue_reoptimize(best_classes, time_limit=2.0)
        reopt_score = calc_score(reopt)
        if reopt_score > best_score:
            best_classes = reopt
            best_score = reopt_score
    
    # Run multiple SA rounds with different parameters
    sa_configs = [
        (1.0, 0.999),
        (0.5, 0.998),
        (0.3, 0.997),
    ]
    
    for temp, cooling in sa_configs:
        if _time_left() < 15.0:
            break
        
        # Polish current best
        polished = hill_climb(best_classes, time_limit=2.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
        
        # SA from current best
        sa_result, sa_score = sa_polish(best_classes, initial_temp=temp, cooling=cooling)
        if sa_score > best_score:
            best_classes = sa_result
            best_score = sa_score
        
        # Residue reoptimization after SA
        if _time_left() > 15.0:
            reopt = residue_reoptimize(best_classes, time_limit=1.5)
            reopt_score = calc_score(reopt)
            if reopt_score > best_score:
                best_classes = reopt
                best_score = reopt_score
        
        # Final polish
        if _time_left() > 10.0:
            polished = hill_climb(best_classes, time_limit=2.0)
            pol_score = calc_score(polished)
            if pol_score > best_score:
                best_classes = polished
                best_score = pol_score
    
    # Final hill climb
    if _time_left() > 5.0:
        polished = hill_climb(best_classes, time_limit=3.0)
        pol_score = calc_score(polished)
        if pol_score > best_score:
            best_classes = polished
            best_score = pol_score
    
    return best_classes
# EVOLVE-BLOCK-END
