import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s kill
    
    # Try n from 34 downwards
    for n in range(34, 1, -1):
        if time.time() > deadline - 5.0:
            break
            
        # Precomputation for this n
        edges = list(combinations(range(n), 3))
        m = len(edges)
        edge_to_idx = {e: i for i, e in enumerate(edges)}
        
        quads = list(combinations(range(n), 4))
        pents = list(combinations(range(n), 5))
        
        edge_to_quads = [[] for _ in range(m)]
        for q_idx, q in enumerate(quads):
            q_edges = list(combinations(q, 3))
            for e in q_edges:
                e_idx = edge_to_idx[e]
                edge_to_quads[e_idx].append(q_idx)
                
        edge_to_pents = [[] for _ in range(m)]
        for p_idx, p in enumerate(pents):
            p_edges = list(combinations(p, 3))
            for e in p_edges:
                e_idx = edge_to_idx[e]
                edge_to_pents[e_idx].append(p_idx)

        # Algebraic seed: use GF(2) structure to generate a low-violation starting point
        # We represent each vertex v as a bit vector and define colors via algebraic rules
        # For n up to 40, we can use 6 bits (2^6 = 64 > 40)
        bits_needed = (n - 1).bit_length() if n > 1 else 1
        
        # Generate algebraic seed
        def algebraic_seed(n, bits_needed):
            """Generate a coloring based on algebraic structure in GF(2)"""
            colors = [0] * m
            for i, (a, b, c) in enumerate(edges):
                # Convert vertices to bit representations
                va, vb, vc = a, b, c
                # Compute XOR-based features
                xor_ab = va ^ vb
                xor_bc = vb ^ vc
                xor_ac = va ^ vc
                # Use parity of certain bit positions as color
                # This creates algebraic structure that may avoid certain patterns
                color = (xor_ab & xor_bc & xor_ac) & 1
                # Add some variation based on vertex sums
                color ^= (va + vb + vc) & 1
                colors[i] = color
            return colors
        
        best_violations = float('inf')
        best_colors = None
        
        # Run local search with algebraic seed initialization
        search_deadline = min(deadline - 5.0, time.time() + 8.0)
        
        # First, try the algebraic seed directly
        colors = algebraic_seed(n, bits_needed)
        
        # Calculate initial violations
        quad_red_counts = [0] * len(quads)
        pent_blue_counts = [0] * len(pents)
        
        for i, c in enumerate(colors):
            if c == 0: # Red
                for q_idx in edge_to_quads[i]:
                    quad_red_counts[q_idx] += 1
            else: # Blue
                for p_idx in edge_to_pents[i]:
                    pent_blue_counts[p_idx] += 1
                    
        violations = sum(1 for c in quad_red_counts if c == 4) + sum(1 for c in pent_blue_counts if c == 10)
        
        if violations == 0:
            return {'n': n, 'colors': colors}
            
        if violations < best_violations:
            best_violations = violations
            best_colors = colors[:]
            
        # Local search loop with violation-focused moves
        local_deadline = min(search_deadline, time.time() + 1.0)
        while time.time() < local_deadline:
            # Focus on edges that are part of violating structures
            # Build list of candidate edges to flip
            candidates = []
            for i in range(m):
                # Check if this edge is in a violating quad or pent
                for q_idx in edge_to_quads[i]:
                    if quad_red_counts[q_idx] == 4:
                        candidates.append(i)
                        break
                else:
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 10:
                            candidates.append(i)
                            break
            
            if not candidates:
                # No obvious violations, try random
                i = random.randint(0, m - 1)
            else:
                i = random.choice(candidates)
            
            old_c = colors[i]
            new_c = 1 - old_c
            
            # Calculate delta violations
            delta_v = 0
            
            if old_c == 0: # Red -> Blue
                for q_idx in edge_to_quads[i]:
                    if quad_red_counts[q_idx] == 4:
                        delta_v -= 1
                for p_idx in edge_to_pents[i]:
                    if pent_blue_counts[p_idx] == 9:
                        delta_v += 1
            else: # Blue -> Red
                for q_idx in edge_to_quads[i]:
                    if quad_red_counts[q_idx] == 3:
                        delta_v += 1
                for p_idx in edge_to_pents[i]:
                    if pent_blue_counts[p_idx] == 10:
                        delta_v -= 1
            
            if delta_v <= 0:
                colors[i] = new_c
                if new_c == 0: # Now Red
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] -= 1
                else: # Now Blue
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] -= 1
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] += 1
                        
                violations += delta_v
                
                if violations == 0:
                    return {'n': n, 'colors': colors}
                    
                if violations < best_violations:
                    best_violations = violations
                    best_colors = colors[:]
            else:
                # Occasionally accept worsening moves to escape local minima
                if random.random() < 0.05:
                    colors[i] = new_c
                    if new_c == 0:
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] += 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] -= 1
                    else:
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] -= 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] += 1
                    violations += delta_v
        
        # If stuck, try random restarts with perturbation of algebraic seed
        while time.time() < search_deadline:
            # Perturb the algebraic seed
            colors = algebraic_seed(n, bits_needed)
            # Apply random flips
            for _ in range(m // 10):
                i = random.randint(0, m - 1)
                colors[i] ^= 1
            
            # Calculate violations
            quad_red_counts = [0] * len(quads)
            pent_blue_counts = [0] * len(pents)
            
            for i, c in enumerate(colors):
                if c == 0:
                    for q_idx in edge_to_quads[i]:
                        quad_red_counts[q_idx] += 1
                else:
                    for p_idx in edge_to_pents[i]:
                        pent_blue_counts[p_idx] += 1
                        
            violations = sum(1 for c in quad_red_counts if c == 4) + sum(1 for c in pent_blue_counts if c == 10)
            
            if violations == 0:
                return {'n': n, 'colors': colors}
                
            if violations < best_violations:
                best_violations = violations
                best_colors = colors[:]
                
            # Quick local search
            quick_deadline = min(search_deadline, time.time() + 0.3)
            while time.time() < quick_deadline:
                candidates = []
                for i in range(m):
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 4:
                            candidates.append(i)
                            break
                    else:
                        for p_idx in edge_to_pents[i]:
                            if pent_blue_counts[p_idx] == 10:
                                candidates.append(i)
                                break
                
                if not candidates:
                    i = random.randint(0, m - 1)
                else:
                    i = random.choice(candidates)
                
                old_c = colors[i]
                new_c = 1 - old_c
                
                delta_v = 0
                if old_c == 0:
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 4:
                            delta_v -= 1
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 9:
                            delta_v += 1
                else:
                    for q_idx in edge_to_quads[i]:
                        if quad_red_counts[q_idx] == 3:
                            delta_v += 1
                    for p_idx in edge_to_pents[i]:
                        if pent_blue_counts[p_idx] == 10:
                            delta_v -= 1
                
                if delta_v <= 0:
                    colors[i] = new_c
                    if new_c == 0:
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] += 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] -= 1
                    else:
                        for q_idx in edge_to_quads[i]:
                            quad_red_counts[q_idx] -= 1
                        for p_idx in edge_to_pents[i]:
                            pent_blue_counts[p_idx] += 1
                            
                    violations += delta_v
                    
                    if violations == 0:
                        return {'n': n, 'colors': colors}
                        
                    if violations < best_violations:
                        best_violations = violations
                        best_colors = colors[:]

    # If no n >= 2 found, return n=1 (trivially valid)
    return {'n': 1, 'colors': []}
