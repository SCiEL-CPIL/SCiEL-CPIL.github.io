"""Improved program: local search with incremental violation counting for hypergraph Ramsey R(4,5;3)."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
from collections import defaultdict

def construct():
    """
    Construct a 2-coloring of triples on n vertices with no red K4^(3) and no blue K5^(3).
    Uses local search (simulated annealing) with incremental violation tracking.
    """
    start_time = time.time()
    deadline = start_time + 115  # safety margin before 150s kill
    
    # Precompute all triples in lexicographic order
    def get_triples(n):
        return list(itertools.combinations(range(n), 3))
    
    # Precompute for each triple, which 4-sets and 5-sets it belongs to
    def build_incidence(n):
        triples = get_triples(n)
        triple_index = {t: i for i, t in enumerate(triples)}
        
        # For each 4-set, list the 4 triples it contains
        quad_triples = {}
        for quad in itertools.combinations(range(n), 4):
            qt = []
            for triple in itertools.combinations(quad, 3):
                qt.append(triple_index[triple])
            quad_triples[quad] = qt
        
        # For each 5-set, list the 10 triples it contains
        quint_triples = {}
        for quint in itertools.combinations(range(n), 5):
            qt = []
            for triple in itertools.combinations(quint, 3):
                qt.append(triple_index[triple])
            quint_triples[quint] = qt
        
        # Reverse: for each triple, which quads and quints contain it
        triple_to_quads = defaultdict(list)
        for quad, qt in quad_triples.items():
            for ti in qt:
                triple_to_quads[ti].append(quad)
        
        triple_to_quints = defaultdict(list)
        for quint, qt in quint_triples.items():
            for ti in qt:
                triple_to_quints[ti].append(quint)
        
        return triples, triple_index, quad_triples, quint_triples, triple_to_quads, triple_to_quints
    
    def local_search(n, max_time=60.0):
        """Run local search to find valid coloring for n vertices."""
        ls_start = time.time()
        ls_deadline = ls_start + max_time
        
        triples, triple_index, quad_triples, quint_triples, triple_to_quads, triple_to_quints = build_incidence(n)
        num_triples = len(triples)
        num_quads = len(quad_triples)
        num_quints = len(quint_triples)
        
        # Initialize with random coloring
        colors = [random.randint(0, 1) for _ in range(num_triples)]
        
        # Compute initial violation counts
        # red_count[quad] = number of red triples in this quad (0-4), violation if ==4
        # blue_count[quint] = number of blue triples in this quint (0-10), violation if ==10
        red_counts = [0] * num_quads
        blue_counts = [0] * num_quints
        quad_idx = {q: i for i, q in enumerate(quad_triples.keys())}
        quint_idx = {q: i for i, q in enumerate(quint_triples.keys())}
        
        # Build list of quad indices and quint indices for fast access
        quad_list = list(quad_triples.keys())
        quint_list = list(quint_triples.keys())
        
        for ti in range(num_triples):
            color = colors[ti]
            for quad in triple_to_quads[ti]:
                qi = quad_idx[quad]
                if color == 0:  # red
                    red_counts[qi] += 1
            for quint in triple_to_quints[ti]:
                qi = quint_idx[quint]
                if color == 1:  # blue
                    blue_counts[qi] += 1
        
        # Count violations
        red_violations = sum(1 for c in red_counts if c == 4)
        blue_violations = sum(1 for c in blue_counts if c == 10)
        total_violations = red_violations + blue_violations
        
        if total_violations == 0:
            return colors
        
        best_colors = colors[:]
        best_violations = total_violations
        
        # Simulated annealing
        temp = 5.0
        cooling = 0.9995
        min_temp = 0.01
        
        while time.time() < ls_deadline:
            if temp < min_temp:
                temp = min_temp
            
            # Pick a random triple to flip
            ti = random.randint(0, num_triples - 1)
            old_color = colors[ti]
            new_color = 1 - old_color
            
            # Compute delta in violations
            delta_red = 0
            delta_blue = 0
            
            for quad in triple_to_quads[ti]:
                qi = quad_idx[quad]
                old_rc = red_counts[qi]
                if old_color == 0:  # was red, now blue
                    # red count decreases by 1
                    if old_rc == 4:
                        delta_red -= 1  # was violation, now not
                    elif old_rc == 3:
                        delta_red += 0  # still not violation
                    # new red count = old_rc - 1
                else:  # was blue, now red
                    if old_rc == 3:
                        delta_red += 1  # becomes violation
                    # new red count = old_rc + 1
            
            for quint in triple_to_quints[ti]:
                qi = quint_idx[quint]
                old_bc = blue_counts[qi]
                if old_color == 1:  # was blue, now red
                    if old_bc == 10:
                        delta_blue -= 1
                    # new blue count = old_bc - 1
                else:  # was red, now blue
                    if old_bc == 9:
                        delta_blue += 1
                    # new blue count = old_bc + 1
            
            delta_total = delta_red + delta_blue
            
            # Accept if improvement or with probability exp(-delta/temp)
            if delta_total <= 0 or random.random() < pow(2.718281828, -delta_total / temp):
                # Apply flip
                colors[ti] = new_color
                
                for quad in triple_to_quads[ti]:
                    qi = quad_idx[quad]
                    if old_color == 0:
                        red_counts[qi] -= 1
                    else:
                        red_counts[qi] += 1
                
                for quint in triple_to_quints[ti]:
                    qi = quint_idx[quint]
                    if old_color == 1:
                        blue_counts[qi] -= 1
                    else:
                        blue_counts[qi] += 1
                
                total_violations += delta_total
                
                if total_violations < best_violations:
                    best_violations = total_violations
                    best_colors = colors[:]
                
                if total_violations == 0:
                    return colors
            
            temp *= cooling
        
        return best_colors if best_violations < total_violations else colors
    
    # Try n from 34 down to 10
    for n in range(34, 9, -1):
        if time.time() > deadline - 10:
            break
        
        max_time_for_n = min(50.0, deadline - time.time() - 5)
        if max_time_for_n < 5.0:
            break
        
        # Try multiple restarts
        found = None
        num_restarts = 3 if n <= 30 else 2
        
        for restart in range(num_restarts):
            if time.time() > deadline - 10:
                break
            
            remaining = deadline - time.time() - 10
            if remaining < 5.0:
                break
            
            t_per_restart = min(remaining / (num_restarts - restart), max_time_for_n / num_restarts)
            if t_per_restart < 2.0:
                break
            
            result = local_search(n, max_time=t_per_restart)
            
            # Verify the result
            if result is not None:
                # Quick verification
                triples = get_triples(n)
                triple_index = {t: i for i, t in enumerate(triples)}
                
                # Check red K4
                valid = True
                for quad in itertools.combinations(range(n), 4):
                    all_red = True
                    for triple in itertools.combinations(quad, 3):
                        if result[triple_index[triple]] != 0:
                            all_red = False
                            break
                    if all_red:
                        valid = False
                        break
                
                if valid:
                    # Check blue K5
                    for quint in itertools.combinations(range(n), 5):
                        all_blue = True
                        for triple in itertools.combinations(quint, 3):
                            if result[triple_index[triple]] != 1:
                                all_blue = False
                                break
                        if all_blue:
                            valid = False
                            break
                
                if valid:
                    found = result
                    break
        
        if found is not None:
            return {'n': n, 'colors': found}
    
    # Fallback: return a small valid coloring
    n = 9
    triples = list(itertools.combinations(range(n), 3))
    colors = [0 if (a + b + c) % 3 == 0 else 1 for a, b, c in triples]
    return {'n': n, 'colors': colors}

def _check_validity(n, colors):
    """Check if coloring has no red K4^(3) and no blue K5^(3)."""
    import itertools
    
    triple_to_color = {}
    idx = 0
    for a, b, c in itertools.combinations(range(n), 3):
        triple_to_color[(a, b, c)] = colors[idx]
        idx += 1
    
    for quad in itertools.combinations(range(n), 4):
        all_red = True
        for triple in itertools.combinations(quad, 3):
            if triple_to_color.get(triple, 1) != 0:
                all_red = False
                break
        if all_red:
            return False
    
    for quint in itertools.combinations(range(n), 5):
        all_blue = True
        for triple in itertools.combinations(quint, 3):
            if triple_to_color.get(triple, 0) != 1:
                all_blue = False
                break
        if all_blue:
            return False
    
    return True
# EVOLVE-BLOCK-END
