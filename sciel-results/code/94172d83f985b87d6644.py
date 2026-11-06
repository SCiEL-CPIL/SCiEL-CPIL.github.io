    for v in verts:
        if _is_input(v, dag):
            ops.append(["load", v])
        else:
            ops.append(["compute", v])
        if v in outs:
            ops.append(["store", v])
    