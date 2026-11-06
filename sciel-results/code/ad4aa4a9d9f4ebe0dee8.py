for p in ps:
    if not try_load(p):
        return False
# Now compute v
...
ops.append(['compute', v])
fast_mem.add(v)
return True
