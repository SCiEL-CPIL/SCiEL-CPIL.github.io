def _apply_nni(tree, target, new_subtree):
    # Recursively replace 'target' with 'new_subtree'
    if tree == target:
        return new_subtree
    if isinstance(tree, tuple):
        left, right = tree
        new_left = _apply_nni(left, target, new_subtree)
        new_right = _apply_nni(right, target, new_subtree)
        return (new_left, new_right)
    return tree
