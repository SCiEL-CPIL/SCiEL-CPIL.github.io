import time
import random
import itertools

N = 16

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    time_limit = 55.0  # Safety margin for 60s limit
    
    # Total edges
    E = N * (N - 1) // 2
    edges = [(i, j) for i in range(N) for j in range(i + 1, N)]
    
    # Map edge tuple to index
    edge_idx = {e: i for i, e in enumerate(edges)}
    
    # Initialize colors
    colors = [0] * E
    used_colors = {0}
    
    # Precompute for each edge, the list of (other_edge_idx1, other_edge_idx2, other_edge_idx3)
    # such that they form a 4-cycle with the current edge.
    # Cycle: (u, v, x, y) -> edges (u,v), (v,x), (x,y), (y,u)
    # For edge (u,v), other edges are (v,x), (x,y), (y,u)
    # Note: (v,x) is same as (x,v). (y,u) is same as (u,y).
    
    neighbor_cycles = [[] for _ in range(E)]
    for i, (u, v) in enumerate(edges):
        for x in range(N):
            if x == u or x == v:
                continue
            for y in range(N):
                if y == u or y == v or y == x:
                    continue
                # Cycle u-v-x-y-u
                e1 = (v, x) if v < x else (x, v)
                e2 = (x, y) if x < y else (y, x)
                e3 = (y, u) if y < u else (u, y)
                
                idx1 = edge_idx[e1]
                idx2 = edge_idx[e2]
                idx3 = edge_idx[e3]
                
                neighbor_cycles[i].append((idx1, idx2, idx3))
                
    def is_valid_change(edge_i, new_color):
        """Check if changing edge_i to new_color creates any rainbow 4-cycle."""
        old_color = colors[edge_i]
        colors[edge_i] = new_color
        
        valid = True
        for idx1, idx2, idx3 in neighbor_cycles[edge_i]:
            c1 = colors[idx1]
            c2 = colors[idx2]
            c3 = colors[idx3]
            c0 = new_color
            
            if len({c0, c1, c2, c3}) == 4:
                valid = False
                break
        
        if not valid:
            colors[edge_i] = old_color
        return valid

    def try_repair(edge_i, new_color):
        """
        Attempt to make the change edge_i -> new_color valid by recoloring other edges in violating cycles.
        Returns True if successful, False otherwise.
        """
        old_color = colors[edge_i]
        colors[edge_i] = new_color
        
        # Find all violating cycles
        violations = []
        for idx1, idx2, idx3 in neighbor_cycles[edge_i]:
            c1 = colors[idx1]
            c2 = colors[idx2]
            c3 = colors[idx3]
            c0 = new_color
            if len({c0, c1, c2, c3}) < 4:
                continue
            # It is a violation. We need to change one of idx1, idx2, idx3 to match c0, c1, c2, or c3.
            # Specifically, to break the rainbow, we can set one edge to a color already present in the cycle.
            # Candidates for new color of idx1: {c0, c2, c3}
            # Candidates for new color of idx2: {c0, c1, c3}
            # Candidates for new color of idx3: {c0, c1, c2}
            
            # We will try to fix this specific violation.
            # But fixing one might break another.
            # Simple strategy: Pick one of the 3 edges randomly, and change it to one of the other 3 colors randomly.
            # Then re-check ALL cycles for edge_i.
            # If all clear, we are good.
            # If not, we might have introduced new violations.
            # Since we are doing local search, we can just try a few random repairs.
            
            violations.append((idx1, idx2, idx3))
            
        if not violations:
            return True
            
        # Try to repair
        # We will pick a random violating cycle and try to fix it.
        # To keep it simple and fast, we try a limited number of random repairs.
        # If we fail, we revert.
        
        # Actually, a safer approach:
        # If we change edge_i, we only need to worry about cycles containing edge_i.
        # If we change one of the other edges (say idx1), it might affect other cycles NOT containing edge_i.
        # However, those other cycles were valid before. Changing idx1 might make them invalid.
        # So "repair" is risky.
        
        # Alternative: Only accept the change if it is valid immediately.
        # This is "greedy" and might get stuck.
        # But with 120 colors and N=16, maybe we can just do random restarts or simple hill climbing?
        
        # Let's try a slightly more powerful move:
        # If direct change fails, try to change edge_i to new_color AND one other edge in a conflicting cycle to a matching color.
        
        for _ in range(10): # Try 10 random repair strategies
            # Pick a random violation
            idx1, idx2, idx3 = random.choice(violations)
            c1 = colors[idx1]
            c2 = colors[idx2]
            c3 = colors[idx3]
            c0 = new_color
            
            # Try to fix by changing idx1
            if random.random() < 0.33:
                target_color = random.choice([c0, c2, c3])
                if target_color == c1: continue
                old_c1 = colors[idx1]
                colors[idx1] = target_color
                # Check if this fixed the specific cycle and didn't break edge_i's other cycles
                # It might break other cycles involving idx1.
                # Let's just check if edge_i is now valid.
                if is_valid_change(edge_i, new_color):
                     # We need to ensure idx1 change didn't break other cycles.
                     # This is complex. 
                     # Let's revert and just use the simple "valid only" move for now, 
                     # but with random restarts.
                     pass 
                colors[idx1] = old_c1
            # ... similar for idx2, idx3
            
        # Given the complexity and time limit, let's stick to a robust simple hill climber:
        # 1. Try to assign a NEW color to a random edge.
        # 2. If valid, keep it.
        # 3. If invalid, try to assign an EXISTING color to a random edge (neutral move) to explore space.
        # 4. Occasionally, allow a "repair" move if we can find one quickly.
        
        # Let's revert the change for this function and implement the simpler loop in the main body.
        colors[edge_i] = old_color
        return False

    # Main Loop
    best_score = len(used_colors)
    best_colors = colors[:]
    
    # Random seed for reproducibility in case of ties, but usually we want randomness.
    # random.seed(42) # Uncomment for debugging
    
    while time.time() - start_time < time_limit:
        # 1. Try to add a new color
        if len(used_colors) < 120:
            # Pick a random edge
            e_idx = random.randint(0, E - 1)
            # Pick a random new color
            available = [c for c in range(120) if c not in used_colors]
            if available:
                new_c = random.choice(available)
                if is_valid_change(e_idx, new_c):
                    used_colors.add(new_c)
                    if len(used_colors) > best_score:
                        best_score = len(used_colors)
                        best_colors = colors[:]
                    continue # Accept and loop again
        
        # 2. Neutral move: Change color to an existing one
        e_idx = random.randint(0, E - 1)
        current_c = colors[e_idx]
        # Pick a random existing color different from current
        if len(used_colors) > 1:
            existing = list(used_colors)
            existing.remove(current_c)
            if existing:
                new_c = random.choice(existing)
                if is_valid_change(e_idx, new_c):
                    # Score doesn't change, but state does
                    pass
        
        # 3. If stuck, try a random restart? Or just keep going.
        # With N=16, the space is small enough that hill climbing might work well.
        
    return best_colors
# EVOLVE-BLOCK-END
