import os, glob, inspect, importlib
try:
    import autoresearch_suite.science_tcs_expansion as m
    print("MODULE FILE:", getattr(m, "__file__", "?"))
    print("DIR:", [x for x in dir(m) if not x.startswith("_")])
    if hasattr(m, "pebble_baseline"):
        print(inspect.getsource(m.pebble_baseline))
except Exception as e:
    print("ERR", e)
