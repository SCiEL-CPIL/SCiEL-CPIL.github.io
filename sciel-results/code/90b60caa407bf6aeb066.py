    node_to_remove = stack[-1][0] # The node we are leaving
    stack.pop()
    path.pop() # Removes node_to_remove
    visited_set.discard(node_to_remove)
    