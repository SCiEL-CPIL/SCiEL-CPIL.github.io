"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


def priority(instance):
    times = np.asarray(instance["times"], dtype=float)
    predecessors = instance["predecessors"]
    n_tasks = len(times)
    
    # Build successor list
    # predecessors[i] is a list of task indices that must precede i
    successors = [[] for _ in range(n_tasks)]
    in_degree = np.zeros(n_tasks, dtype=int)
    
    for i in range(n_tasks):
        for p in predecessors[i]:
            successors[p].append(i)
            in_degree[i] += 1
            
    # Calculate Bottom Level (Critical Path Length to End)
    # BL[i] = times[i] + max(BL[j] for j in successors[i])
    # We need to process tasks in reverse topological order (sinks first).
    # We can use a queue for tasks with no successors (out-degree 0).
    
    out_degree = np.array([len(s) for s in successors], dtype=int)
    queue = [i for i in range(n_tasks) if out_degree[i] == 0]
    # To ensure deterministic and efficient processing, we don't strictly need a specific order
    # as long as we process a node only after all its successors have been processed.
    # A simple stack or queue works.
    
    bottom_level = np.zeros(n_tasks, dtype=float)
    
    # Use a stack for DFS-like post-order or a queue for BFS-like reverse processing.
    # Since we need max of successors, and successors are "downstream",
    # we can process from the end of the line backwards.
    
    # Let's use a simple list as a stack.
    # Initialize with tasks having no successors.
    stack = queue.copy()
    
    # We need to know when all successors of a node are processed.
    # We can track the number of unprocessed successors.
    unprocessed_successors = out_degree.copy()
    
    while stack:
        node = stack.pop()
        # All successors of 'node' have already been processed (by definition of how we push)
        # Wait, the standard topological sort pushes nodes when in-degree is 0.
        # Here we want reverse topological sort.
        # A node is ready to be processed if all its successors have been processed.
        # So, we maintain a count of unprocessed successors.
        # Initially, tasks with 0 successors are ready.
        
        # When we process a task 'node', we reduce the unprocessed count of its predecessors.
        
        max_succ_bl = 0.0
        # Actually, we don't need to look at successors again if we store the max.
        # But we haven't calculated it yet.
        
        # Let's refine the algorithm:
        # 1. out_degree[i] = number of successors.
        # 2. ready_queue = tasks with out_degree == 0.
        # 3. While ready_queue not empty:
        #      pop task u
        #      if u has no successors: BL[u] = times[u]
        #      else: BL[u] = times[u] + max(BL[v] for v in successors[u])
        #           -- Wait, this requires BL[v] to be known.
        #           Since v is a successor, v was pushed to ready_queue before u?
        #           No. If v is a successor of u, v is "downstream".
        #           We process downstream first.
        #           So v is processed before u.
        #           So when we pop u, all its successors v have already been popped and their BL calculated.
        
        # So:
        # 1. Initialize unprocessed_successors[i] = out_degree[i]
        # 2. ready = [i for i if unprocessed_successors[i] == 0]
        # 3. While ready:
        #      u = ready.pop()
        #      max_succ = 0.0
        #      for v in successors[u]:
        #          if bottom_level[v] > max_succ:
        #              max_succ = bottom_level[v]
        #      bottom_level[u] = times[u] + max_succ
        #      for p in predecessors[u]:
        #          unprocessed_successors[p] -= 1
        #          if unprocessed_successors[p] == 0:
        #              ready.append(p)
              
    # Re-implementing carefully to avoid logic errors in the thought process above.
    # The logic:
    # A task is "ready" for BL calculation when all its successors are calculated.
    # Successors are "downstream".
    # So we start with tasks that have NO successors.
    # When we calculate BL for task u, we decrement the "pending successors count" for all predecessors of u.
    # If a predecessor's pending count reaches 0, it becomes ready.
    
    out_deg = np.array([len(s) for s in successors], dtype=int)
    pending_succs = out_deg.copy()
    ready = [i for i in range(n_tasks) if pending_succs[i] == 0]
    bl = np.zeros(n_tasks, dtype=float)
    
    while ready:
        u = ready.pop()
        max_succ_bl = 0.0
        for v in successors[u]:
            if bl[v] > max_succ_bl:
                max_succ_bl = bl[v]
        bl[u] = times[u] + max_succ_bl
        
        for p in predecessors[u]:
            pending_succs[p] -= 1
            if pending_succs[p] == 0:
                ready.append(p)
                
    return bl
# EVOLVE-BLOCK-END
