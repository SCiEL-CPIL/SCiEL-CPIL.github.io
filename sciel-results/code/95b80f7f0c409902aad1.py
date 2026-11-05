"""Size-efficient sorting network search on 16 wires using Vectorized SA."""
import time
import random
import numpy as np

N = 16
# Deadline: 90s limit, keep 10s safety margin for cleanup/verification
DEADLINE = time.time() + 80.0

# Precompute all 65536 boolean inputs as a matrix (65536, 16)
# bits[i, k] is the k-th bit of i
BITS = np.zeros((1 << N, N), dtype=np.int8)
for k in range(N):
    BITS[:, k] = (np.arange(1 << N) >> k) & 1

def verify_network_batch(network):
    """
    Verify a network against all 65536 inputs using NumPy vectorization.
    Returns True if valid, False otherwise.
    """
    if not network:
        return N == 1
    
    # Copy input matrix
    wires = BITS.copy()
    
    for i, j in network:
        # Check if wire[i] > wire[j]
        # If wires[i] is 1 and wires[j] is 0, swap
        mask = (wires[:, i] > wires[:, j])
        # Perform swap
        # temp = wires[i]
        # wires[i] = wires[j]
        # wires[j] = temp
        # Using numpy swap or simple assignment
        # Since mask is boolean, we can use np.where or swap
        # np.swapaxes doesn't work on 1D slices easily in this context, 
        # so we do:
        # tmp = wires[:, i].copy()
        # wires[:, i] = np.where(mask, wires[:, j], wires[:, i])
        # wires[:, j] = np.where(mask, tmp, wires[:, j])
        
        # More efficient:
        # Only swap where mask is True
        # We can just swap the columns in the masked rows
        # But numpy doesn't support fancy indexing swap directly without temp
        # Let's use temp
        tmp = wires[:, i].copy()
        wires[:, i] = np.where(mask, wires[:, j], wires[:, i])
        wires[:, j] = np.where(mask, tmp, wires[:, j])

    # Check if sorted
    # wires[:, k] <= wires[:, k+1] for all k
    # If any wire[k] > wire[k+1], it's not sorted
    # diff = wires[:, :-1] - wires[:, 1:]
    # return not np.any(diff < 0)
    
    # Faster check:
    # A sorted binary vector has no 10 pattern
    # So we check if there exists k such that wires[k]=1 and wires[k+1]=0
    for k in range(N - 1):
        if np.any(wires[:, k] > wires[:, k + 1]):
            return False
            
    return True

def odd_even_network():
    """Odd-even transposition sorting network: 120 comparators."""
    return [(i, i + 1) for phase in range(N)
            for i in range(phase % 2, N - 1, 2)]

def compute_depth(network):
    """Compute parallel depth."""
    wire_time = [0] * N
    for i, j in network:
        t = max(wire_time[i], wire_time[j]) + 1
        wire_time[i] = t
        wire_time[j] = t
    return max(wire_time)

def random_neighbor(network):
    """
    Generate a random neighbor by modifying the network.
    1. Remove a random comparator (40%)
    2. Add a random comparator (30%)
    3. Move a random comparator (30%)
    """
    if not network:
        return []
    
    new_net = list(network)
    r = random.random()
    
    if r < 0.4 and len(new_net) > 1:
        # Remove
        idx = random.randint(0, len(new_net) - 1)
        new_net.pop(idx)
    elif r < 0.7:
        # Add
        i = random.randint(0, N - 2)
        j = random.randint(i + 1, N - 1)
        pos = random.randint(0, len(new_net))
        new_net.insert(pos, (i, j))
    else:
        # Move
        if len(new_net) > 1:
            idx = random.randint(0, len(new_net) - 1)
            comp = new_net.pop(idx)
            pos = random.randint(0, len(new_net))
            new_net.insert(pos, comp)
            
    return new_net

def crossover(net1, net2):
    """
    Genetic crossover: take prefix from net1 and suffix from net2.
    """
    if not net1 or not net2:
        return list(net1) if net1 else list(net2)
    
    # Ensure lengths are similar for better crossover
    # Random split point
    split = random.randint(0, min(len(net1), len(net2)))
    child = net1[:split] + net2[split:]
    return child

def construct():
    start_time = time.time()
    
    # Initialize with baseline
    best_net = odd_even_network()
    best_score = len(best_net) # 120
    current_net = list(best_net)
    current_score = best_score
    
    # Simulated Annealing Parameters
    # We want to minimize comparator count
    # Score = len(network)
    # We accept worse scores with probability exp(-delta / T)
    
    T = 10.0 # Initial temperature
    T_min = 0.1
    cooling_rate = 0.995
    
    # Population for genetic algorithm (optional, but SA is primary)
    # Let's stick to SA + Random Restarts if stuck
    
    stuck_counter = 0
    
    while time.time() < DEADLINE:
        # Generate neighbor
        neighbor = random_neighbor(current_net)
        neighbor_score = len(neighbor)
        
        # Verify neighbor
        # Optimization: If neighbor_score >= current_score, we might still accept it if T is high
        # But if neighbor_score > current_score + 5, it's unlikely to be useful, skip verification?
        # No, we need to know if it's valid.
        
        # Quick check: if neighbor is invalid, reject
        # To save time, we can check validity only if it has potential to be better
        # Or if we are accepting worse states.
        
        is_valid = verify_network_batch(neighbor)
        
        if is_valid:
            if neighbor_score < current_score:
                # Accept
                current_net = neighbor
                current_score = neighbor_score
                
                if current_score < best_score:
                    best_net = list(current_net)
                    best_score = current_score
                    stuck_counter = 0
                    # Cool down faster when improving? Or just standard
                else:
                    stuck_counter += 1
                    
            else:
                # Worse or equal
                # Accept with probability
                delta = neighbor_score - current_score
                if delta == 0:
                    # Equal length, accept with some probability to explore
                    if random.random() < 0.5:
                        current_net = neighbor
                else:
                    prob = np.exp(-delta / T)
                    if random.random() < prob:
                        current_net = neighbor
                        current_score = neighbor_score
                stuck_counter += 1
        else:
            # Invalid, reject
            stuck_counter += 1
            
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
            
        # Random Restart if stuck too long
        if stuck_counter > 500:
            # Try a random restart from a random valid network or baseline
            # Generate a random sparse network
            # Or just reset to best_net and increase T slightly
            current_net = list(best_net)
            current_score = best_score
            T = max(T, 5.0)
            stuck_counter = 0
            
            # Occasionally try a completely random small network
            if random.random() < 0.1:
                # Try to build a small random network
                # This is risky but might find a local optimum
                pass # Keeping it simple for now, just restart from best

    # Final verification of best_net
    if not verify_network_batch(best_net):
        # Fallback
        best_net = odd_even_network()
        
    return best_net

# EVOLVE-BLOCK-START
# The construct function is defined above, but the task requires keeping the function names.
# The prompt says "Keep the function names and signatures of the seed program."
# Seed program had: construct()
# I have defined construct().
# I should ensure the EVOLVE-BLOCK-START/END markers are respected if they were in the seed.
# The seed had:
# # EVOLVE-BLOCK-START
# def construct():
#    ...
# # EVOLVE-BLOCK-END
# I will place the final construct definition inside the block if I were modifying, 
# but since I am providing the complete program, I will just ensure the structure is valid.
# The prompt asks for a COMPLETE program.
# EVOLVE-BLOCK-END
