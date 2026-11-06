"""
Communication-minimizing DAG pebbling.

The naive schedule (which scores 1.0) typically computes each vertex in topological
order, storing the result in fast memory, and then either keeping it or deleting it
when its parents are done. This is equivalent to a simple forward pass.

We can do better by observing that for a layered DAG, the optimal strategy often
involves recomputing values rather than storing them, if the fast memory budget is
small. This is the classic "pebbling" problem.

However, the task says "Score mean transfers / naive schedule". So if we use the
same number of transfers as the naive schedule, we get 1.0. To get a score > 1.0,
we need to use FEWER transfers than the naive schedule.

The naive schedule likely does:
1. Load each input vertex (1 transfer per input).
2. For each internal vertex, load its parents (if not in memory), compute, store.
3. Store each output vertex (1 transfer per output).
4. Delete intermediate vertices as needed to free memory.

If the fast memory budget is large enough to hold all vertices, the naive schedule
is optimal and we can't do better. But if the budget is small, we must recompute.

Wait, the score is "mean transfers / naive schedule". So if we use fewer transfers,
the ratio is < 1.0, and the search score is seed/value = 1.0 / (transfers/naive) = naive/transfers.
So if we use fewer transfers, the search score is > 1.0.

But the task says "Score mean transfers / naive schedule on three layered DAGs; exact state-machine verification."
And "The evaluator measures the raw value (lower is better). The search score is seed/value, so 1.0 equals the seed program and above 1.0 is an improvement".

So the raw value is the mean of (transfers/naive) over the three DAGs. Lower is better for the raw value.
The search score is 1.0 / raw_value. So if raw_value < 1.0, search score > 1.0.

So we need to minimize the total number of transfers (load + store + delete + compute? No, "transfers" likely means load/store/delete, not compute).

Let's assume "transfers" = number of load/store/delete operations.

The naive schedule's transfer count is the baseline. We need to beat it.

Strategy:
1. Parse the DAG to understand its structure.
2. If the fast memory budget is large enough to hold all "live" vertices, use a simple topological sort and keep everything in memory.
3. Otherwise, use a recursive strategy that minimizes transfers.

However, implementing a full optimal pebbling algorithm is complex. Let's start with a simple heuristic:
- If the number of vertices <= fast_memory_budget, just do a topological sort and keep everything.
- Otherwise, use a divide-and-conquer approach on the layers.

But we need to be careful about the time limit. Let's implement a simple version first and see if it works.

Actually, the seed program imports from `autoresearch_suite.science_tcs_expansion import pebble_baseline`. We don't have access to that, so we need to implement it from scratch.

Let's assume the DAG is a dict or a list of vertices with parents. The task says "dag has topologically ordered parents and output vertices".

Let's assume the dag is a list of vertices, each with a list of parents. The vertices are indexed 0 to n-1, and they are in topological order. The output vertices are the last few vertices, or they are specified.

Wait, the task says "Inputs start blue, all outputs must finish blue". So the inputs are the vertices with no parents, and the outputs are the vertices that must be blue at the end.

Let's assume the dag is a list of lists, where dag[i] is the list of parents of vertex i. The vertices are in topological order, so dag[i] only contains indices < i.

The fast_memory_budget is the maximum number of red values that can be in fast memory at any time.

Operations:
- load: bring a blue value from slow memory to fast memory (1 transfer).
- store: move a red value from fast memory to slow memory (1 transfer).
- compute: compute a red value from its parents in fast memory (0 transfers).
- delete: remove a red value from fast memory (1 transfer).

All inputs start blue in slow memory. All outputs must finish blue in slow memory.

The naive schedule likely does:
1. Load all inputs.
2. For each vertex in topological order:
   a. If the vertex is not in fast memory, compute it (if its parents are in fast memory) or load it (if it's an input).
   b. If the vertex's children are all computed, delete it to free memory.
3. Store all outputs.

But this is not necessarily optimal.

Let's implement a simple version that does a topological sort and keeps as many vertices in memory as possible.

We'll use a greedy algorithm:
- Process vertices in topological order.
- Keep track of which vertices are in fast memory.
- For each vertex, if its parents are in fast memory, compute it. Otherwise, we need to load or recompute.
- If the fast memory is full, delete a vertex that is no longer needed (i.e., all its children have been processed).

But this is still complex. Let's start with a very simple version that just does a topological sort and assumes we can keep everything in memory. If the budget is too small, we'll just do our best.

Actually, the task says "at most 20000 operations". So we can't do too many operations.

Let's implement a simple version and see if it works.

We'll assume the dag is a list of lists of parent indices.

We'll also assume that the output vertices are the last k vertices, where k is the number of vertices with no children.

Let's start coding.
"""

import time
import sys

def pebble(dag, fast_memory_budget):
    start_time = time.time()
    deadline = start_time + 80  # 80 seconds to be safe
    
    # Parse the DAG
    # dag is a list of lists, where dag[i] is the list of parents of vertex i
    n = len(dag)
    
    # Find the output vertices (vertices with no children)
    # First, compute the children of each vertex
    children = [[] for _ in range(n)]
    for i in range(n):
        for p in dag[i]:
            children[p].append(i)
    
    # Output vertices are those with no children
    outputs = [i for i in range(n) if len(children[i]) == 0]
    
    # Input vertices are those with no parents
    inputs = [i for i in range(n) if len(dag[i]) == 0]
    
    # Simple strategy: topological sort, keep everything in memory if possible
    # If the budget is large enough, just load all inputs, compute all, store all outputs
    
    # Check if we can fit all vertices in fast memory
    if n <= fast_memory_budget:
        # Just do a simple topological sort
        ops = []
        in_memory = set()
        
        # Load all inputs
        for v in inputs:
            ops.append(['load', v])
            in_memory.add(v)
        
        # Compute all other vertices in topological order
        for v in range(n):
            if v not in in_memory:
                # Check if all parents are in memory
                if all(p in in_memory for p in dag[v]):
                    ops.append(['compute', v])
                    in_memory.add(v)
                else:
                    # This shouldn't happen in a topological sort
                    pass
        
        # Store all outputs
        for v in outputs:
            if v in in_memory:
                ops.append(['store', v])
                in_memory.remove(v)
        
        return ops
    
    # If we can't fit everything, use a simpler strategy
    # Just do a topological sort and delete vertices as soon as they're no longer needed
    ops = []
    in_memory = set()
    # Count how many times each vertex is needed (i.e., how many children it has that haven't been processed)
    remaining_children = [len(children[i]) for i in range(n)]
    
    # Load all inputs
    for v in inputs:
        if len(in_memory) < fast_memory_budget:
            ops.append(['load', v])
            in_memory.add(v)
    
    # Process vertices in topological order
    for v in range(n):
        # Check if all parents are in memory
        parents_in_memory = all(p in in_memory for p in dag[v])
        
        if not parents_in_memory:
            # Need to load or recompute parents
            # For now, just load them (this is not optimal)
            for p in dag[v]:
                if p not in in_memory:
                    if len(in_memory) >= fast_memory_budget:
                        # Need to delete something
                        # Find a vertex that is no longer needed
                        for d in in_memory:
                            if remaining_children[d] == 0:
                                ops.append(['delete', d])
                                in_memory.remove(d)
                                break
                    ops.append(['load', p])
                    in_memory.add(p)
        
        # Compute v
        ops.append(['compute', v])
        in_memory.add(v)
        
        # Decrement the remaining children count of v's parents
        for p in dag[v]:
            remaining_children[p] -= 1
        
        # If v is an output, store it
        if v in outputs:
            ops.append(['store', v])
            in_memory.remove(v)
    
    return ops

# EVOLVE-BLOCK-START
# The actual pebble function is defined above
# EVOLVE-BLOCK-END
