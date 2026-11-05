def do_reversal(order, i, j):
    # order is the permutation of vertices
    # reverse order[i:j+1]
    order[i:j+1] = order[i:j+1][::-1]
