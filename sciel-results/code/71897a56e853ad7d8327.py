    relevant = set(outputs)
    stack = list(outputs)
    while stack:
        v = stack.pop()
        for p in parents_dict.get(v, []):
            if p not in relevant:
                relevant.add(p)
                stack.append(p)
