import time
import random
import numpy as np
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 120s limit

    # Fallback: n=9 modular coloring (valid)
    def mod_color(n, mod=3):
        return [0 if (a + b + c) % mod == 0 else 1 for a, b, c in combinations(range(n), 3)]

    best_n = 9
    best_colors = mod_color(9, 3)

    def is_valid(n, colors):
        """Verify a coloring is valid (no red K4, no blue K5)."""
        cm = {}
        for i, t in enumerate(combinations(range(n), 3)):
            cm[t] = colors[i]
        for quad in combinations(range(n), 4):
            a, b, c, d = quad
            if cm[(a,b,c)] == 0 and cm[(a,b,d)] == 0 and cm[(a,c,d)] == 0 and cm[(b,c,d)] == 0:
                return False
        for quint in combinations(range(n), 5):
            all_blue = True
            for t in combinations(quint, 3):
                if cm[t] != 1:
                    all_blue = False
                    break
            if all_blue:
                return False
        return True

    def build_structures(target_n):
        """Precompute all structural mappings for a given n."""
        triple_list = list(combinations(range(target_n), 3))
        num_triples = len(triple_list)
        triple_to_idx = {t: i for i, t in enumerate(triple_list)}

        all_quads = list(combinations(range(target_n), 4))
        quad_triples_idx = []
        for a, b, c, d in all_quads:
            quad_triples_idx.append([
                triple_to_idx[(a,b,c)], triple_to_idx[(a,b,d)],
                triple_to_idx[(a,c,d)], triple_to_idx[(b,c,d)]
            ])
        num_quads = len(quad_triples_idx)

        all_quints = list(combinations(range(target_n), 5))
        quint_triples_idx = []
        for q5 in all_quints:
            quint_triples_idx.append([triple_to_idx[t] for t in combinations(q5, 3)])
        num_quints = len(quint_triples_idx)

        # Inverse mappings: for each triple, which quads/quints contain it
        # We store lists of indices for fast iteration
        triple_in_quads = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quad_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quads[tidx].append(qi)

        triple_in_quints = [[] for _ in range(num_triples)]
        for qi, qt in enumerate(quint_triples_idx):
            for pos, tidx in enumerate(qt):
                triple_in_quints[tidx].append(qi)

        quad_triples_np = np.array(quad_triples_idx, dtype=np.int32)
        quint_triples_np = np.array(quint_triples_idx, dtype=np.int32)

        return {
            'triple_list': triple_list,
            'num_triples': num_triples,
            'quad_triples_np': quad_triples_np,
            'quint_triples_np': quint_triples_np,
            'num_quads': num_quads,
            'num_quints': num_quints,
            'triple_in_quads': triple_in_quads,
            'triple_in_quints': triple_in_quints,
        }

    def compute_initial_state(color_arr, structs):
        """Compute violation counts and near-violation counts using vectorized numpy."""
        quad_colors = structs['quad_triples_np']
        quad_red_counts = np.sum(color_arr[quad_colors] == 0, axis=1)
        rk4 = int(np.sum(quad_red_counts == 4))
        near_rk4 = int(np.sum(quad_red_counts == 3))

        quint_colors = structs['quint_triples_np']
        quint_blue_counts = np.sum(color_arr[quint_colors] == 1, axis=1)
        bk5 = int(np.sum(quint_blue_counts == 10))
        near_bk5 = int(np.sum(quint_blue_counts == 9))

        # Total weighted energy
        energy = 100 * (rk4 + bk5) + 10 * (near_rk4 + near_bk5)
        
        return energy, rk4, bk5, near_rk4, near_bk5, quad_red_counts, quint_blue_counts

    def generate_initial_coloring(target_n, restart_idx, num_triples, triple_list):
        """Generate diverse initial colorings."""
        if restart_idx == 0:
            return np.array([0 if (a+b+c) % 3 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 1:
            return np.array([0 if (a+b+c) % 4 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 2:
            return np.array([0 if (a+b+c) % 5 == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
        elif restart_idx == 3:
            return np.random.randint(0, 2, size=num_triples, dtype=np.int8)
        elif restart_idx == 4:
            arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            arr[np.random.random(num_triples) < 0.4] = 0
            return arr
        elif restart_idx == 5:
            arr = np.random.randint(0, 2, size=num_triples, dtype=np.int8)
            arr[np.random.random(num_triples) < 0.4] = 1
            return arr
        else:
            base_mod = 3 + (restart_idx % 3)
            base = np.array([0 if (a+b+c) % base_mod == 0 else 1 for (a,b,c) in triple_list], dtype=np.int8)
            flip_mask = np.random.random(num_triples) < 0.2
            base[flip_mask] = 1 - base[flip_mask]
            return base

    def local_search_penalized(color_arr, structs, max_iters, deadline, temp_start=5.0, temp_end=0.1):
        """
        Local search using penalized energy (violations + near-violations).
        Returns (colors, rk4, bk5).
        """
        num_triples = structs['num_triples']
        triple_in_quads = structs['triple_in_quads']
        triple_in_quints = structs['triple_in_quints']
        quad_triples_np = structs['quad_triples_np']
        quint_triples_np = structs['quint_triples_np']

        energy, rk4, bk5, near_rk4, near_bk5, quad_red_counts, quint_blue_counts = compute_initial_state(color_arr, structs)

        if rk4 == 0 and bk5 == 0:
            return color_arr, rk4, bk5

        it = 0
        while it < max_iters:
            if time.time() > deadline - 2:
                break
            if rk4 == 0 and bk5 == 0:
                break

            temp = temp_start * (temp_end / temp_start) ** (it / max_iters)

            # Choose a triple to flip
            # Bias towards triples in violating or near-violating structures
            tidx = -1
            r = random.random()
            
            # 50% chance to pick from red violations/near-violations
            if r < 0.5:
                # Find quads with 3 or 4 reds
                # quad_red_counts is a numpy array
                candidates = np.where((quad_red_counts == 3) | (quad_red_counts == 4))[0]
                if len(candidates) > 0:
                    qi = candidates[random.randint(0, len(candidates)-1)]
                    qt = quad_triples_np[qi]
                    tidx = int(qt[random.randint(0, 3)])
            # 30% chance to pick from blue violations/near-violations
            elif r < 0.8:
                candidates = np.where((quint_blue_counts == 9) | (quint_blue_counts == 10))[0]
                if len(candidates) > 0:
                    qi = candidates[random.randint(0, len(candidates)-1)]
                    qt = quint_triples_np[qi]
                    tidx = int(qt[random.randint(0, 9)])
            
            # 20% chance random
            if tidx == -1:
                tidx = random.randint(0, num_triples - 1)

            old_color = int(color_arr[tidx])
            new_color = 1 - old_color
            
            delta_energy = 0

            # Process Quads
            for qi in triple_in_quads[tidx]:
                current_red_count = int(quad_red_counts[qi])
                
                if old_color == 0: # tidx is Red, becoming Blue
                    # Red count decreases by 1
                    new_red_count = current_red_count - 1
                    
                    # Old contribution
                    if current_red_count == 4:
                        delta_energy -= 100
                    elif current_red_count == 3:
                        delta_energy -= 10
                    
                    # New contribution
                    if new_red_count == 4:
                        delta_energy += 100
                    elif new_red_count == 3:
                        delta_energy += 10
                        
                    # Update count
                    quad_red_counts[qi] = new_red_count

                else: # tidx is Blue, becoming Red
                    # Red count increases by 1
                    new_red_count = current_red_count + 1
                    
                    # Old contribution
                    if current_red_count == 4:
                        delta_energy -= 100
                    elif current_red_count == 3:
                        delta_energy -= 10
                    
                    # New contribution
                    if new_red_count == 4:
                        delta_energy += 100
                    elif new_red_count == 3:
                        delta_energy += 10
                        
                    # Update count
                    quad_red_counts[qi] = new_red_count

            # Process Quints
            for qi in triple_in_quints[tidx]:
                current_blue_count = int(quint_blue_counts[qi])
                
                if old_color == 1: # tidx is Blue, becoming Red
                    # Blue count decreases by 1
                    new_blue_count = current_blue_count - 1
                    
                    # Old contribution
                    if current_blue_count == 10:
                        delta_energy -= 100
                    elif current_blue_count == 9:
                        delta_energy -= 10
                    
                    # New contribution
                    if new_blue_count == 10:
                        delta_energy += 100
                    elif new_blue_count == 9:
                        delta_energy += 10
                        
                    # Update count
                    quint_blue_counts[qi] = new_blue_count

                else: # tidx is Red, becoming Blue
                    # Blue count increases by 1
                    new_blue_count = current_blue_count + 1
                    
                    # Old contribution
                    if current_blue_count == 10:
                        delta_energy -= 100
                    elif current_blue_count == 9:
                        delta_energy -= 10
                    
                    # New contribution
                    if new_blue_count == 10:
                        delta_energy += 100
                    elif new_blue_count == 9:
                        delta_energy += 10
                        
                    # Update count
                    quint_blue_counts[qi] = new_blue_count

            # Acceptance criterion
            if delta_energy <= 0:
                color_arr[tidx] = new_color
                energy += delta_energy
                
                # Update global counters
                # We need to update rk4, bk5, near_rk4, near_bk5
                # This is tricky because we updated the arrays. 
                # Instead of updating counters incrementally (error prone), 
                # we can just recompute if energy is low or periodically?
                # No, recomputing is O(N^3). 
                # Let's update counters incrementally too.
                
                # Re-evaluate rk4/bk5/near_rk4/near_bk5 from the arrays is too slow per step.
                # We can track them:
                # But we modified quad_red_counts. 
                # Let's just rely on the final check for validity.
                # For the loop condition, we can check if energy == 0.
                # But we need rk4/bk5 for the return and the break condition.
                # Actually, if energy == 0, then rk4=bk5=near_rk4=near_bk5=0.
                # So we can just check energy.
                
            else:
                if random.random() < np.exp(-delta_energy / max(temp, 1e-10)):
                    color_arr[tidx] = new_color
                    energy += delta_energy
                else:
                    # Revert changes to counts
                    for qi in triple_in_quads[tidx]:
                        if old_color == 0:
                            quad_red_counts[qi] += 1
                        else:
                            quad_red_counts[qi] -= 1
                    for qi in triple_in_quints[tidx]:
                        if old_color == 1:
                            quint_blue_counts[qi] += 1
                        else:
                            quint_blue_counts[qi] -= 1

            it += 1

        # Final check for actual violations
        rk4, bk5, _, _, _, _, _ = compute_initial_state(color_arr, structs)
        
        return color_arr, rk4, bk5

    for target_n in range(10, 35):
        if time.time() > deadline - 8:
            break

        # Precompute structures
        structs = build_structures(target_n)
        triple_list = structs['triple_list']
        num_triples = structs['num_triples']

        found = False
        max_restarts = 200
        for restart in range(max_restarts):
            if time.time() > deadline - 3:
                break

            # Initialize coloring
            color_arr = generate_initial_coloring(target_n, restart, num_triples, triple_list)

            # Check if already valid
            energy, rk4, bk5, _, _, _, _ = compute_initial_state(color_arr, structs)
            if rk4 == 0 and bk5 == 0:
                found = True
                break

            # Local search with penalized energy
            max_iters = 50000
            color_arr, rk4, bk5 = local_search_penalized(color_arr, structs, max_iters, deadline)

            if rk4 == 0 and bk5 == 0:
                found = True
                break

        if found:
            # Verify the solution
            colors_list = color_arr.tolist()
            if is_valid(target_n, colors_list):
                best_n = target_n
                best_colors = colors_list
            else:
                # Should not happen, but break if invalid
                break
        else:
            # If we failed to find a valid coloring for this n,
            # it's unlikely we can find one for larger n quickly
            if target_n >= 30:
                break

    return {'n': best_n, 'colors': best_colors}
