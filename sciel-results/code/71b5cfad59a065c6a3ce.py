"""
Flow-shop permutation construction using Tabu Search with Or-opt moves.
Strategy:
1. Initialize with NEH heuristic for a strong starting point.
2. Use Tabu Search with Or-opt block moves (size 1-3).
3. Maintain a tabu list to prevent cycling and use aspiration criteria to escape local optima.
4. Vectorized makespan calculation using NumPy for speed.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

def _makespan_vec(order, P):
    """
    Compute makespan for a given job order using vectorized NumPy operations.
    P: (n_jobs, n_machines)
    order: list/array of job indices
    """
    if len(order) == 0:
        return 0.0
    
    n_jobs_seq = len(order)
    n_machines = P.shape[1]
    
    # Extract processing times for the sequence
    seq_P = P[order]
    
    # C_machines[j] will hold the completion time of job j on the current machine.
    c_machines = np.zeros(n_machines)
    last_completion = 0.0
    
    for j in range(n_jobs_seq):
        row = seq_P[j]
        c_curr = 0.0
        for m in range(n_machines):
            c_curr = max(c_machines[m], c_curr) + row[m]
            c_machines[m] = c_curr
        last_completion = c_curr
        
    return last_completion

def _neh_heuristic(P):
    """
    NEH heuristic for initial solution.
    """
    n_jobs, n_machines = P.shape
    totals = P.sum(axis=1)
    order_indices = np.argsort(-totals)
    
    current_seq = []
    for idx in order_indices:
        best_pos = 0
        best_makespan = float('inf')
        
        for pos in range(len(current_seq) + 1):
            new_seq = current_seq[:pos] + [idx] + current_seq[pos:]
            ms = _makespan_vec(new_seq, P)
            if ms < best_makespan:
                best_makespan = ms
                best_pos = pos
        
        current_seq.insert(best_pos, idx)
        
    return current_seq

def _generate_or_opt_neighbor(seq):
    """
    Generate a random Or-opt neighbor by moving a block of size 1, 2, or 3.
    Returns (new_seq, move_info) where move_info is a tuple identifying the move for tabu list.
    """
    n = len(seq)
    if n <= 3:
        # Just swap or random shuffle for small N
        new_seq = seq.copy()
        if n > 1:
            i, j = random.sample(range(n), 2)
            new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
            return new_seq, (i, j, 'swap')

    # Choose block size
    max_block = min(3, n - 1)
    block_size = random.randint(1, max_block)
    
    # Choose start position
    start = random.randint(0, n - block_size)
    
    # Extract block
    block = seq[start:start + block_size]
    
    # Remove block
    remaining = seq[:start] + seq[start + block_size:]
    
    # Choose insertion position in remaining
    # Avoid inserting back at the same place
    # Valid positions: 0 to len(remaining)
    # If ins_pos == start, it's the original position.
    # If ins_pos == start + 1, it's immediately after the removed block? 
    # Actually, if we remove [start:start+bs], the element at start in 'remaining' 
    # is the original start+bs. 
    # Inserting at 'start' puts it back.
    # Inserting at 'start+1' puts it after the first element that was originally after the block? No.
    # Let's just filter out 'start' to be safe.
    
    valid_positions = [p for p in range(len(remaining) + 1) if p != start]
    if not valid_positions:
         # Fallback
         i, j = random.sample(range(n), 2)
         new_seq = seq.copy()
         new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
         return new_seq, (i, j, 'swap')
         
    ins_pos = random.choice(valid_positions)
    
    new_seq = remaining[:ins_pos] + block + remaining[ins_pos:]
    
    # Move info: (block_content_tuple, start_pos, ins_pos)
    # To check tabu, we need to know if the reverse move is tabu.
    # A move is defined by moving block B from pos S to pos I.
    # The reverse move is moving block B from pos I to pos S.
    # We store (tuple(block), start, ins_pos)
    
    move_info = (tuple(block), start, ins_pos)
    
    return new_seq, move_info

def _is_tabu(move_info, tabu_list):
    """
    Check if the move is tabu.
    A move (block, start, ins) is tabu if its reverse (block, ins, start) is in the tabu list.
    Note: The tabu list stores the moves that were just made. So if we made move M, 
    the reverse of M is forbidden.
    So we check if (block, ins_pos, start) is in tabu_list.
    """
    block, start, ins_pos = move_info
    reverse_move = (block, ins_pos, start)
    return reverse_move in tabu_list

def order_jobs(processing_times):
    """
    Order jobs to minimize makespan using NEH initialization and Tabu Search.
    """
    P = np.asarray(processing_times)
    if P.size == 0:
        return []
        
    n_jobs, n_machines = P.shape
    if n_jobs == 0:
        return []
    if n_jobs == 1:
        return [0]
        
    start_time = time.time()
    # 150s limit, use 145s to be safe
    time_limit = 145.0
    
    # 1. Initialize with NEH
    current_seq = _neh_heuristic(P)
    current_makespan = _makespan_vec(current_seq, P)
    
    best_seq = current_seq.copy()
    best_makespan = current_makespan
    
    # Tabu Search Parameters
    tabu_tenure = 10  # Length of tabu list
    max_iter = 100000
    i = 0
    
    # Use a list for tabu moves, maintaining order (oldest first)
    tabu_list = []
    
    while (time.time() - start_time) < time_limit and i < max_iter:
        i += 1
        
        # Generate neighbor
        new_seq, move_info = _generate_or_opt_neighbor(current_seq)
        new_makespan = _makespan_vec(new_seq, P)
        
        delta = new_makespan - current_makespan
        
        # Check tabu status
        is_tabu = _is_tabu(move_info, tabu_list)
        
        # Aspiration criterion: accept tabu move if it improves global best
        if is_tabu and new_makespan >= best_makespan:
            # Reject this move, try another one
            continue
            
        # Accept the move
        current_seq = new_seq
        current_makespan = new_makespan
        
        if current_makespan < best_makespan:
            best_makespan = current_makespan
            best_seq = current_seq.copy()
            
        # Update tabu list
        tabu_list.append(move_info)
        if len(tabu_list) > tabu_tenure:
            tabu_list.pop(0)
            
        # Dynamic tenure adjustment?
        # If we are stuck (no improvement in last X iterations), increase tenure
        # For simplicity, keep static tenure but maybe increase it slightly if problem is large
        if n_jobs > 50 and i % 1000 == 0:
             tabu_tenure = min(20, tabu_tenure + 1)

    return best_seq
# EVOLVE-BLOCK-END
