"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def partition(adjacency):
    """
    Solve Max-Cut using Simulated Annealing with vectorized move evaluation.
    """
    n = len(adjacency)
    if n == 0:
        return np.array([], dtype=np.int8)
    
    # Convert to numpy array for speed
    adj = np.array(adjacency, dtype=np.int8)
    
    # Precompute degrees to help with cut calculation updates
    # Actually, we can just maintain the cut value incrementally.
    
    start_time = time.time()
    deadline = start_time + 148.0  # Safety margin before 150s limit
    
    best_labels = None
    best_cut = -1
    
    # Simulated Annealing Parameters
    # T0: Initial temperature
    # T_end: Final temperature
    # alpha: Cooling rate
    # We want to run for a fixed number of steps or until time limit.
    
    # Estimate cost per step. Vectorized step is O(N).
    # N=100. 100 operations is very fast.
    # We can do millions of steps.
    
    # Let's try a fixed number of restarts with SA, or just one long SA run?
    # SA with restarts is usually better for Max-Cut.
    
    # Let's do multiple restarts.
    # Each restart:
    # 1. Random init
    # 2. Run SA for some time budget or steps
    
    # Strategy:
    # Use the entire time budget for SA.
    # Periodically restart if stuck? Or just one long run with high T start?
    # Given N=100, one long run might get stuck.
    # Let's do 5 restarts, splitting the time budget.
    
    num_restarts = 5
    time_per_restart = (deadline - start_time) / num_restarts
    
    for r in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Random initialization
        labels = np.random.randint(0, 2, size=n)
        
        # Calculate initial cut
        # Cut = sum(adj[i,j] * (1 if labels[i] != labels[j] else 0))
        # Vectorized: 
        # diff = labels[:, None] != labels[None, :]
        # cut = np.sum(adj * diff)
        # But we need to update this efficiently.
        
        # Efficient update:
        # When flipping vertex v:
        # Old contribution of edges connected to v:
        # If label[v] was 0:
        #   Edges to 0 were 0 (not cut), edges to 1 were 1 (cut).
        #   New label 1:
        #   Edges to 0 become 1 (cut), edges to 1 become 0 (not cut).
        #   Delta = (edges to 0) - (edges to 1)
        # Let's precompute row sums or use matrix multiplication.
        
        # Let's maintain a vector `gain` where gain[v] is the change in cut if we flip v.
        # gain[v] = sum(adj[v, u] * (2*labels[u] - 1) * (1 if labels[v]==0 else -1)) ?
        # Let's derive:
        # Cut = 0.5 * sum_{i,j} adj[i,j] * (1 - (2*labels[i]-1)*(2*labels[j]-1))
        # Let s[i] = 2*labels[i] - 1 (spin). s[i] in {-1, 1}.
        # Cut = 0.5 * (Total_Edges - sum_{i,j} adj[i,j] s[i] s[j])
        # Maximizing Cut is equivalent to Minimizing E = sum_{i,j} adj[i,j] s[i] s[j].
        # This is the Ising model ground state.
        
        # Let's work with spins s.
        # s = 2*labels - 1
        # E = s^T A s
        # If we flip s[v] -> -s[v]:
        # Delta E = E_new - E_old
        # E_new = (s - 2 s[v] e_v)^T A (s - 2 s[v] e_v)
        # = s^T A s - 2 s[v] s^T A e_v - 2 s[v] e_v^T A s + 4 s[v]^2 e_v^T A e_v
        # = E - 4 s[v] (A s)[v] + 0 (since A[v,v]=0 usually, and s[v]^2=1)
        # Wait, (A s)[v] is the sum of neighbors' spins weighted by adj.
        # So Delta E = -4 s[v] * (A s)[v]
        
        # We want to minimize E (maximize Cut).
        # So if Delta E < 0, it's an improvement.
        
        # Let's use this formulation.
        
        s = np.random.choice([-1, 1], size=n)
        
        # Calculate initial E
        # E = s @ (adj @ s)
        # Note: adj is symmetric? The problem says "unweighted 100-node graphs", usually implies undirected.
        # The input is a "bool matrix". We should ensure it's treated as undirected or handle directed?
        # "edges crossing the partition" usually implies undirected.
        # If adj is not symmetric, we should symmetrize it or just use it as is?
        # Standard Max-Cut is on undirected graphs.
        # Let's assume adj[i][j] implies an edge.
        # If it's directed, "crossing" is ambiguous.
        # Given "unweighted ... graphs", I will assume undirected.
        # I'll symmetrize to be safe: adj = np.maximum(adj, adj.T)
        
        adj_sym = np.maximum(adj, adj.T)
        
        # Initial Energy
        As = adj_sym @ s
        E = s @ As
        
        # Initial Cut
        # Total edges M = sum(adj_sym) / 2
        M = np.sum(adj_sym) // 2
        # Cut = M - 0.5 * E
        # Actually, E = sum s_i s_j w_ij.
        # If s_i != s_j, term is -1. If s_i == s_j, term is 1.
        # E = sum (same) - sum (diff) = (M - C) - C = M - 2C.
        # So C = (M - E) / 2.
        
        current_cut = (M - E) / 2.0
        
        if current_cut > best_cut:
            best_cut = current_cut
            best_labels = ((s + 1) // 2).astype(np.int8)
            
        # SA Loop
        # We need to update E efficiently.
        # When flipping v:
        # Delta E = -4 * s[v] * (As[v])
        # New As:
        # As_new = As - 2 * s[v] * adj_sym[:, v]
        
        # Temperature schedule
        # Let's run for a fixed number of steps based on time.
        # Or just loop until time runs out for this restart.
        
        T = 1.0
        T_min = 0.01
        # Estimate steps we can do.
        # One step:
        # 1. Random v
        # 2. Calc Delta E
        # 3. Accept/Reject
        # 4. Update s, As, E
        # This is very fast.
        
        steps = 0
        while time.time() < start_time + (r+1)*time_per_restart:
            # Pick random vertex
            v = np.random.randint(n)
            
            # Calculate Delta E
            delta_E = -4.0 * s[v] * As[v]
            
            # Accept if delta_E < 0 (improvement)
            # Or with probability exp(-delta_E / T) if delta_E > 0
            if delta_E < 0:
                # Always accept improvement
                accept = True
            else:
                # Accept with probability exp(-delta_E / T)
                # delta_E is positive, so -delta_E is negative
                # exp(negative) < 1
                # Use log for stability?
                # log_prob = -delta_E / T
                # if log(rand) < log_prob: accept
                # rand > 0
                # log(rand) is in (-inf, 0]
                # -delta_E/T is negative.
                # So we accept if log(rand) < -delta_E/T
                # i.e. rand < exp(-delta_E/T)
                
                # Optimization: if T is very small, exp is 0.
                if T < 1e-6:
                    accept = False
                else:
                    # Use random() for speed
                    accept = np.random.random() < np.exp(-delta_E / T)
            
            if accept:
                # Flip s[v]
                old_s_v = s[v]
                s[v] = -old_s_v
                
                # Update As
                # As = As - 2 * old_s_v * adj_sym[:, v]
                # Note: adj_sym[:, v] is a vector.
                As -= 2.0 * old_s_v * adj_sym[:, v]
                
                # Update E
                E += delta_E
                
                # Update current_cut
                current_cut = (M - E) / 2.0
                
                if current_cut > best_cut:
                    best_cut = current_cut
                    best_labels = ((s + 1) // 2).astype(np.int8)
            
            # Cooling
            # Geometric cooling is standard.
            # But we don't know total steps in advance.
            # Let's use a fixed cooling factor per step.
            # T = T * 0.999
            # This might cool too fast or too slow.
            # Alternative: Linear cooling based on time?
            # Or just keep T constant for a while, then drop?
            
            # Let's try: T starts at 1.0, decreases by 0.999 each step.
            # If we do 10k steps, T ~ 0.004.
            # If we do 100k steps, T ~ 0.
            
            T *= 0.999
            if T < T_min:
                T = T_min
            
            steps += 1
    
    if best_labels is None:
        return np.arange(n) % 2
    return best_labels
# EVOLVE-BLOCK-END
